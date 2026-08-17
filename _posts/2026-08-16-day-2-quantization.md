---
title: "Quantization for LLM Inference: What INT8 and INT4 Really Do, and a Measured Benchmark"
date: 2026-08-16
categories: [llm, inference, optimization, quantization]
permalink: /day-2/
---


By Yuvraj Singh Bhadoria

Quantization is the part of inference that sounds like a math trick but is really a memory trick. In the last post we measured that decode is memory bound, not compute bound, and that the weights plus the KV cache are what fill GPU memory. This post asks the obvious next question. What if the weights themselves took less memory? I am writing these as I learn, and this time I actually ran the numbers on a T4 instead of trusting the marketing. By the end you should be able to read any quantization paper or model card and know exactly what is being claimed.

## 0. The question

When you serve a model, every token forces the GPU to read the entire set of weights out of memory. The speed of that step is roughly weight size divided by memory bandwidth. So if we store each weight in 1 byte instead of 2, we move half the data per token, and decode should get faster. That is the promise. The rest of this post tests whether the promise holds, and on what hardware, and at what cost to quality.

## 1. The one number that explains inference cost

A forward pass for one token multiplies the input by the weight matrices. The weights are fixed, so for every token the GPU must read all of them from memory. Time per token is roughly weight size in bytes divided by bandwidth. Tokens per second is roughly bandwidth divided by weight size.

![FP16 weights reduced to INT8 then INT4, showing fewer bits means a smaller model and less data moved from memory](/assets/day2-quantization-overview.png)
*Quantization trades numerical precision for smaller weights and less memory traffic.*

For OPT-125M the weights are 125 million parameters at 2 bytes each, about 250 MB. A T4 has about 320 GB/s of bandwidth, so the best case is 320 divided by 0.25, about 1280 tok/s, if weight reading were the only thing happening. It is not. The previous post measured about 308 tok/s on vLLM and about 37 tok/s on plain transformers. Both sit far below 1280, which tells us that on a model this small, weight reading is not the bottleneck. Overhead is.

My take: name your regime before you optimize. Small model means overhead bound, so quantization will not make it faster. Huge model means memory bound, and quantization is the single biggest lever you have. Everything below is pointless unless you know which one you are in.

## 2. The core idea: weights as marks on a ruler

Quantization stores each weight with fewer bits. The cleanest picture is a ruler.

A weight is a number, say 0.37. FP16 stores it precisely with 16 bits. Quantization treats the range as a ruler with a fixed number of marks. INT8 gets 256 marks, INT4 gets 16. We store which mark is closest, plus one number for the step width. That number is the scale.

![High-precision weights mapped onto a coarse integer ruler using scale and zero point, then reconstructed approximately](/assets/day2-quantization-scale.png)
*Quantization maps continuous values onto a smaller set of integer marks.*

```
scale = max_abs / (2^b - 1)
q     = round( r / scale )     # the mark number we store
r'    = q * scale              # the value we use at inference
```

Here is what each symbol means:

- `b` is the number of bits. INT4 means b = 4, so 2^b - 1 = 15 marks. INT8 means b = 8, so 2^b - 1 = 255 marks. More bits give more marks and a smaller error.
- `max_abs` is the largest absolute weight in the tensor. It sets how wide the ruler is.
- `r` is the real weight, a full precision number like 0.37, that we are compressing.
- `q` is the integer mark we actually store. It is round(r / scale), a small whole number.
- `r'` (read r prime) is the value we get back at inference: q times scale. It is close to r but not exact.
- `scale` is the step width between two marks. It is the one extra number we store next to the marks.

Scale is the step width. Divide the weight by the scale, round to the nearest mark, store q, and at inference multiply q back by scale to get r'. The error between r and r' can never exceed half a step.

A real example makes it stick. Four weights [0.8, -0.2, 0.05, -1.0], INT4, so 15 steps. Biggest absolute value is 1.0, scale = 0.0667.

```
weights = [0.8, -0.2, 0.05, -1.0]
scale   = max(abs(w) for w in weights) / (2**4 - 1)   # 0.0667
q       = [round(w / scale) for w in weights]         # [12, -3, 1, -15]
r_prime = [qi * scale for qi in q]                    # [0.80, -0.20, 0.0667, -1.00]
```

Three come back nearly exact, the smallest lands at 0.0667, error 0.017. Now add one outlier, 50.0. Scale becomes 3.33, 0.8 divided by 3.33 rounds to 0, and the small weights collapse. One outlier destroyed everything. Keep that picture. Every method below stops exactly this.

One refinement real systems use: the ruler need not be symmetric. You can slide it so the first mark sits at the smallest weight, not at minus the largest. That needs a second stored number, the zero point, and is called asymmetric quantization. In symbols, w_q = round(w / scale) + zero_point, and the value used at inference is (w_q - zero_point) * scale. The zero point shifts the ruler so its marks line up with the data's actual minimum. It wastes no marks on empty range. Symmetric quantization, with zero point at 0, is what we used above and is simpler.

The symbols: `w` is the real weight, `w_q` is the stored integer (round(w / scale) plus the zero point), `zero_point` is the shift that lines the first mark up with the data's minimum, and `scale` is the step width from the formula above.

A second refinement is group wise quantization: instead of one scale per whole row, use one scale per small group of 128 columns. More scales, finer fit, and that is how INT4 stays accurate.

My take: if you remember one formula, remember r' = q * scale. Everything else is just deciding where to put the marks and how many scales to keep.

## 3. The precision spectrum

A trained model is usually FP16, 2 bytes per weight. The formats you will meet are points on the bits versus accuracy tradeoff.



- FP32, 4 bytes. Training only, too big to serve.
- FP16, 2 bytes. Normal serving format.
- BF16, 2 bytes but with FP32's range and less precision. Training favorite, close enough to FP16 for serving.
- INT8, 1 byte. Half of FP16, runs on almost any GPU including the T4.
- FP8, 1 byte float. Needs H100 or L40 class hardware. A T4 cannot do it.
- INT4, half a byte. Quarter of FP16. Where most LLM quantization research lives.
- INT2 and 1 bit, research only, large quality loss.

Halve the bits and you roughly halve the memory, which can mean up to 2x the decode ceiling. But only when weight reading is the bottleneck. On a 70B model the weights are 140 GB and reading dominates, so halving is decisive. On OPT-125M it is hidden by overhead.

My take: pick the format your hardware can actually run. A T4 does INT8 and INT4. FP8 needs an H100 or L40. Format envy is wasted if your GPU cannot compute it.

## 4. What actually gets quantized

A forward pass carries three kinds of numbers, not equally easy to shrink.

![A transformer split into weights, activations, and KV cache, three different quantization problems](/assets/day2-what-gets-quantized.png)
*Weights, activations, and KV cache are three separate engineering problems.*

- Weights are fixed after training. This is what people mean by quantization. Compute scales once, reuse forever.
- Activations are fresh every token, range shifts constantly, hard to quantize because of outliers.
- The KV cache is the cached Key and Value from the last post. It grows with context and is the memory hog we found there.

So weights are easy, KV cache is easy-ish, activations are the trap. The rest of this post is how people avoid that trap.

My take: weights are a solved problem. Activations are the unsolved one. Almost all the clever work below is about activations, not weights.

## 5. The outlier problem, and how to fit the ruler

Activations hold a few channels 10 to 100 times larger than the rest. Call them outliers. One global ruler forces a huge scale, and every normal value gets crushed into a single mark. That is why naive rounding fails on transformers.

Why do outliers appear at all? In a transformer the attention layer runs a softmax over the tokens, and a few channels spike when one token dominates that softmax. Those spikes are the outliers. They change for every token, so activations need a scale per token, while weights are fixed and get a scale per row. That mismatch, per token for activations versus per row for weights, is exactly why activations are the hard part and weights are the easy part.

![A single large activation outlier stretches the INT8 range so normal values are crushed into one region](/assets/day2-outliers.png)
*One outlier forces a huge scale, crushing the normal values.*

Two practical fixes come from the ruler picture. Per channel scaling gives each row its own scale, so small rows keep their marks. Without it, quality collapses. Post training quantization, PTQ, computes scales from a few hundred normal sentences, no retraining. That is what vLLM, bitsandbytes, GPTQ, and AWQ all ship.

My take: once this clicks, the field stops looking like tricks. LLM.int8, SmoothQuant, GPTQ, AWQ are four answers to one question: where do the big values go so the small ones keep their marks? Remember that. It is the spine of everything below.

## 6. LLM.int8: isolate the outliers

LLM.int8, from Dettmers and others in 2022, leaves outlier columns in FP16 and quantizes the normal columns to INT8, then adds the two results. Outliers pulled out first means they no longer inflate the normal scale. It uses vector wise quantization, picking the scale per row of the activation and per group of columns rather than one scale for the whole tensor, so each part keeps its marks. The mixed precision matmul is really two matmuls, one in INT8 and one in FP16, added at the end. Net memory about half, accuracy near FP16. The code is one line.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto",
)
```

![LLM.int8 splits the matrix so normal values take the INT8 path and outlier values stay in FP16, then combines](/assets/day2-llm-int8.png)
*Most values use INT8; rare outliers stay in FP16.*

NF4, Normal Float 4, is a 4 bit format tuned to how weights are distributed, not a uniform ruler. Its 16 marks are placed at the quantiles of the weight distribution, so more marks sit where weights cluster and fewer at the rare edges. It is what QLoRA uses for fine tuning in 4 bit.

My take: LLM.int8 is the honest baseline. It proves the core idea, half the memory with the same quality, using the smallest possible change. Everything after it is about going smaller or faster.

## 7. SmoothQuant, GPTQ, and AWQ

SmoothQuant attacks outliers from the other side. The layer is Y = X times W. For any per channel scale s, X times W equals (X divided by s) times (W times s). So SmoothQuant can shrink the spike in X and move that magnitude into W instead.

Concretely, for each channel it sets s equal to (max of X in that channel, to the power alpha) divided by (max of W in that channel, to the power 1 minus alpha), with alpha around 0.5. The dial alpha decides how much difficulty moves: at 0 all of it stays in X, at 1 all of it moves to W.

The symbols: `X` is the activation for that channel, `W` is the weight for that channel, `s` is the per channel scale SmoothQuant computes, and `alpha` is the dial, about 0.5, that splits the difficulty between activations and weights.

X becomes clean INT8 using a scale per token, W absorbs the variation and quantizes well per channel. Both sides INT8, no FP16 half, no split penalty. SmoothQuant moved the fragile outlier difficulty out of activations and into the calm weights.

![SmoothQuant migrates scaling from activations into weights so the activation distribution becomes easy to quantize to INT8](/assets/day2-smoothquant.png)
*SmoothQuant moves quantization difficulty from activations into weights.*

GPTQ pushes weights to INT4. Its insight: minimize error in the layer output Y, not per weight. It quantizes one column, then immediately adjusts the not yet quantized columns to cancel that column's error in Y. Each mistake is absorbed by columns still ahead. By the end every weight is INT4 but the output stays close to the original. The mental model: paint one strip, fix the spill beside it before moving on.

How does it know each weight's importance to Y? From the calibration data. The sensitivity of Y to a weight is captured by the activation covariance, the matrix X transpose X over the calibration set, which stands in for the Hessian. Weights whose columns matter more to Y get corrected harder. GPTQ also uses group wise quantization, one scale per 128 columns, which is the q_group_size you see in the AWQ code below. The symbols: `q_group_size` is how many columns share one scale (128 here), and the Hessian is the stand in for how sensitive the layer output is to each weight, estimated from the activation covariance X transpose X over a small calibration set.

AWQ agrees outliers matter but protects before quantizing. It finds the salient channels from the calibration activations: a channel is salient if its average input magnitude is large, because multiplying a large input by that weight drives the output.

AWQ scales those salient channels up through the same X/s times W times s identity, so INT4 spends its 16 marks on what matters, then divides the scale back out at inference. It uses group wise quantization, one scale per 128 columns, the same q_group_size as GPTQ. No reconstruction loop. The code is short.

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    safetensors=False,
)
model.quantize(tokenizer, quant_config={"w_bit": 4, "q_group_size": 128})
```

![GPTQ minimizes reconstruction error after quantizing, while AWQ protects salient channels before quantizing, both reaching INT4](/assets/day2-gptq-vs-awq.png)
*GPTQ fixes the error after; AWQ protects the important weights first.*

Put GPTQ and AWQ side by side. GPTQ quantizes first and repairs after. AWQ reshapes so the damage never lands. Same destination, different route.

My take: if you only serve one quantized model, AWQ is the safe default. Fast to apply, no heavy reconstruction, native in vLLM. GPTQ still wins at very low bits sometimes, but for most serving the difference is small and AWQ's simplicity pays off.

![INT4 weight-only inference loads fewer weight bytes from HBM, dequantizes, then multiplies, so compute is not all INT4](/assets/day2-int4-inference.png)
*INT4 cuts weight memory and traffic; the compute is not entirely INT4.*

## 8. Quantizing the KV cache

The KV cache is the direct continuation of the memory story. It grows every token and fills GPU memory first. Quantizing it to INT8 halves that growth.

![FP16 KV cache blocks become smaller INT8 blocks, shrinking per-token memory so more sequences fit](/assets/day2-kv-cache.png)
*A smaller KV cache fits more sequences, but measure the accuracy tradeoff.*

KV values are bounded per token, so per token scaling works, and per head scaling is used so each attention head keeps its own marks. Caution: a small error in K or V shifts the attention score, and attention is sensitive, so KV cache stays at INT8 while weights go to INT4. FP8 KV cache is best but needs H100 or A100. On a T4 the KV cache stays FP16 and only the weights are quantized. Shrink what you can.

My take: quantize the KV cache only after you have measured the quality hit on your own prompts. Attention is unforgiving, so earn the 2x memory win before you trust it.

## 9. My measured benchmark

I ran OPT-125M on a T4 across three configurations with the Hugging Face transformers engine. For each I generated 128 tokens three times and measured tokens per second, real weight memory, total GPU memory, and perplexity on a fixed sentence. Lower perplexity is better. All three rows are the same engine, so they compare methods fairly. Do not chart them against the vLLM 308 tok/s number, that was a different engine with different overhead.

![Measured OPT-125M on T4: weights drop 250 to 166 to 123 MB, perplexity stays near 24 to 25, INT8 is slower because the model is overhead bound](/assets/day2-benchmark.png)
*Smaller weights do not automatically mean faster on a small model.*

| Config | Weights | tok/s | GPU MB | Perplexity | Note |
|---|---|---|---|---|---|
| FP16 baseline | 250 MB | 37.0 | 687 | 24.90 | reference |
| LLM.int8 | 166 MB | 13.9 | 687 | 24.35 | weights halved, quality held |
| INT4 (NF4) | 123 MB | 55.8 | 687 | 25.11 | weights quartered, quality held |

Read it in three passes.

Quality held. Perplexity stays at 24 to 25 across all three. Dropping to half a byte per weight did not break the model. That is the proof the outlier handling works. The tiny row differences are noise from one short sentence, not a trend.

Memory shrank but you cannot see it in the GPU column. Weights fell 250 to 166 to 123 MB, not the clean 125 and 62 you would get if every parameter were 1 or half a byte. Bitsandbytes leaves the embedding and output layer in FP16 and stores scales, so totals are about 166 and 123. The GPU column stays 687 every time because CUDA reserves a few hundred MB of fixed overhead regardless of model size. On a 125M model that overhead hides the saving. On a 70B model the weights are 140 GB and the same halving would show as about 70 GB saved. The win is real, just invisible here.

Speed did the opposite of marketing, and that is the lesson. INT8 dropped to 13.9, slower than FP16. To multiply in INT8 the GPU must dequantize back to FP16 first, an extra step every token. On a 125M model the weights fit in cache, so the bottleneck is that overhead, not bandwidth, and INT8 loses. INT4 quarters the weights, the bandwidth saved beats the dequant cost, and it wins at 55.8. This is the gap from section 1: tok/s equals bandwidth over weight size only when weight reading is the bottleneck. On a toy it is not, so quantization does not speed up decode. On a 70B model it does, for two reasons. First, weight reading dominates, so moving 2x or 4x less data helps. Second, the GPU's Tensor Cores compute INT8 and INT4 matmuls faster than FP16, not just load them faster. Big models win on both bandwidth and compute.

A note on AWQ: the autoawq library does not support OPT-125M, so it has no measured row. It would land near the INT4 row in memory and perplexity, a 4 bit method solving the same outlier problem by protecting salient channels up front.

My take: the number that matters for your career is the memory one. Nobody hires an inference engineer to shave 50 tok/s off a 125M toy. They hire one to fit a 70B model on hardware that should not hold it, or to serve ten times more users on one box. Quantization does that. Speed is a bonus that appears exactly when the model is big enough to need it.

Reproduce it:

```python
import time, torch, math
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

mid = "facebook/opt-125m"
tok = AutoTokenizer.from_pretrained(mid)
pid = tok("The history of artificial intelligence is", return_tensors="pt").input_ids.to("cuda")
wiki = "Artificial intelligence is the field of computer science that studies machines able to perform tasks requiring human intelligence. It includes learning, reasoning, and perception."

cfgs = {
    "FP16":     dict(device_map="auto"),
    "LLM.int8": dict(quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto"),
    "INT4-NF4": dict(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4"), device_map="auto"),
}

for name, kw in cfgs.items():
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(mid, **kw)
    ts = []
    for _ in range(3):
        t0 = time.time()
        o = m.generate(pid, max_new_tokens=128, temperature=0.7, do_sample=True)
        ts.append((o.shape[1] - pid.shape[1]) / (time.time() - t0))
    i = tok(wiki, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        ppl = math.exp(m(i, labels=i).loss.item())
    mb = sum(p.numel() * p.element_size() for p in m.parameters()) / 1e6
    print(f"{name}: {sum(ts)/3:.1f} tok/s | weights {mb:.0f} MB | GPU {torch.cuda.max_memory_allocated()/1e6:.0f} MB | ppl {ppl:.2f}")
```

The printed output:

```
FP16: 37.0 tok/s | weights 250 MB | GPU 687 MB | ppl 24.90
LLM.int8: 13.9 tok/s | weights 166 MB | GPU 687 MB | ppl 24.35
INT4-NF4: 55.8 tok/s | weights 123 MB | GPU 687 MB | ppl 25.11
```

## Takeaway

Quantization is one idea with four fixes.

1. Weights become marks on a ruler, with a per row scale so every row keeps its marks, done after training on a small calibration set.
2. Naive rounding fails because activations hold outliers that blow up the scale. Every real method handles outliers.
3. LLM.int8 isolates outliers in FP16. SmoothQuant moves the difficulty into weights. GPTQ repairs column errors. AWQ protects salient channels up front.
4. The KV cache quantizes too, to INT8, and FP8 is best if your hardware allows it.

The takeaway that survives: quantization is a memory lever, not a speed lever, for models this size. It fits a 70B model on one GPU and keeps quality usable. Measure the memory win, trust the quality hold, expect the speed win only when weight reading is the bottleneck.

## Day 2 summary

![Day 2 summary: from memory bound decode to ruler quantization, outliers, four methods, and a measured benchmark](/assets/day2-quantization-overview.png)

## Next (Day 3)

In the next post we go one level up from the weights into the serving engine, and look at how the scheduler decides which requests run together and how continuous batching keeps the GPU full. That closes the loop from memory bound decode to a full picture of a serving system.

## Visual references

The diagrams in this article are original sketchnote illustrations created for this article. Their technical concepts were informed by the LLM.int8, SmoothQuant, GPTQ, and AWQ papers listed in References, and by the memory bound decode view from the JAX Scaling Book inference chapter (MIT licensed) and NVIDIA inference optimization material.

## References

- Dettmers et al., LLM.int8: 8 bit Matrix Multiplication for Transformers at Scale, 2022.
- Xiao et al., SmoothQuant: Accurate and Efficient Post Training Quantization for LLM, 2023.
- Frantar et al., GPTQ: Accurate Post Training Quantization for Generative Pre trained Transformers, 2023.
- Lin et al., AWQ: Activation aware Weight Quantization for LLM Compression and Acceleration, 2023.
- OPT model card, Meta AI and Hugging Face.
- NVIDIA Tesla T4 datasheet.
