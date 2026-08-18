---
title: "Quantization for LLM Inference: What INT8 and INT4 Really Do, and a Measured Benchmark"
date: 2026-08-16
categories: [llm, inference, optimization, quantization]
permalink: /day-2/
---

By Yuvraj Singh Bhadoria.

The last post was about memory and batching. We learned that serving is mostly a memory problem, and that PagedAttention and continuous batching make memory use efficient. This post goes one level deeper. Instead of managing memory better, we make the model itself smaller. That is quantization.

I am writing these as I learn, measuring everything instead of trusting marketing.

## 0. The question

A model is just a pile of numbers called weights. The bigger the pile, the more memory it takes and the more data the GPU must move to generate each token. So a natural question: can we store those numbers with fewer bits and still get the same answers?

That is quantization. But it is not free. Fewer bits means less precision, and less precision can quietly hurt quality. The job is to throw away bits where it does not matter and keep them where it does.

By the end of this post you should know what INT8 and INT4 actually do, why the naive version fails, how the clever methods fix it, and what my own benchmark on a tiny model showed. Let's go in order.

## 1. Memory bandwidth: the real cost of inference

Here is the whole money question in one sentence. Every time the model writes a new token, it has to read its entire set of weights back out of GPU memory. The weights never change, so the GPU pulls the same big pile of numbers, token after token. The speed is therefore set by one thing: how fast the GPU can move those weights out of memory. That speed is called bandwidth.

![FP16 weights reduced to INT8 then INT4, showing fewer bits means a smaller model and less data moved from memory](/assets/day2-quantization-overview.png)
*Quantization trades numerical precision for smaller weights and less memory traffic.*

Picture a warehouse. The weights are a fixed pile of boxes. To serve one token, you wheel the whole pile from the back room to the counter. A bigger pile takes longer to move. Quantization is simply making the boxes smaller, so each trip carries less.

Tokens per second is roughly bandwidth divided by weight size. Bandwidth is how fast memory moves, a T4 does about 320 GB/s. Weight size is how many bytes the model takes up.

Let me make that real with our toy. OPT-125M weights are 125 million numbers at 2 bytes each, about 250 MB, or 0.25 GB. Divide 320 by 0.25 and the ceiling is about 1280 tok/s, but only if reading weights were the only work. It is not. The last post measured about 308 tok/s on vLLM and 37 tok/s on plain transformers. Both are far below 1280, which tells us that on a model this small, reading weights is not what slows it down. Overhead is.

My take: name your regime before you optimize. On a small model you are overhead bound, so quantization will not make it faster, it only makes it smaller. On a huge model you are memory bound, and shrinking the weights is the single biggest win you have. Everything below is pointless until you know which one you are in.

## 2. The core idea: weights as marks on a ruler

Quantization stores each weight with fewer bits. The cleanest picture is a ruler.

A weight is just a number, say 0.37. To keep it in the computer you have to write it down somehow. FP16 uses 16 bits, which can describe 65536 different numbers, so 0.37 is written almost exactly.

Now imagine we are only allowed 4 bits. Four bits can describe only 16 different numbers. So across the whole spread of weights, our ruler can have only 16 marks. The gap between two marks is the scale. With 16 marks that gap is large, so 0.37 cannot sit exactly on a mark. It rounds to the nearest one, and that small miss is the only error we add. INT8 allows 256 marks, the gap is tiny, and 0.37 lands almost exactly.

So quantization is simple: use a coarser ruler, write down the nearest mark, and keep one extra number for the gap between marks. That extra number is the scale.

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

So the flow is: divide the weight by the scale, round to the nearest mark, store q, and at inference multiply q back by scale to get r'. The error between r and r' can never exceed half a step.

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

My take: if you remember one formula, remember r' = q * scale. Everything else is just deciding where to put the marks and how many scales to keep.

## 3. The precision spectrum

Precision is just how many bits we give each number. More bits, more room to be exact. Fewer bits, smaller and faster to move but rougher.

- FP32, 4 bytes. Training only. Too big to serve.
- FP16, 2 bytes. The normal way to serve. Half precision.
- BF16, 2 bytes. Same size as FP16 but can hold both tiny and huge numbers, so training does not break. For serving it is close to FP16.
- INT8, 1 byte. Half the size of FP16. Now we are quantizing.
- FP8, 1 byte. A format built for AI math, but only on new GPUs like H100 or L40. Most of us cannot use it yet.
- INT4 or NF4, half a byte. A quarter of FP16. This is where the real memory win comes from.

The pattern is the point. Every time you halve the bits, you halve the bytes, so you halve the memory and the data moved. INT8 is half of FP16, INT4 is a quarter. That is the whole game.

My take: pick the format your hardware can actually run. A T4 does INT8 and INT4. FP8 needs an H100 or L40. Format envy is wasted if your GPU cannot compute it.

## 4. What actually gets quantized

Three things in a model hold numbers: the weights, the activations, and the KV cache. They are not equally easy to quantize.

Weights are the easiest. They are fixed after training. You can look at the whole block of weights once, pick scales, and store the small version. Do it once, use it forever.

The KV cache is next. It grows per token, so it behaves a bit like activations, but it has a fixed size per token and per attention head, so one scale per token works well.

Activations are the trap. They are the numbers the model computes as it runs, and they change with every input. You cannot pick the scale ahead of time, because you do not know the input yet. Worse, activations have rare huge spikes, and those spikes break the simple rounding method. Most of the clever work in this post is about activations, not weights.

![Weights are static and easy to quantize, the KV cache is bounded per token, and activations are dynamic with outlier spikes](/assets/day2-what-gets-quantized.png)
*Weights are easy, the KV cache is easy-ish, activations are the trap.*

My take: weights are a solved problem. Activations are the unsolved one. If a method name sounds fancy, it is almost always fighting the activation outliers.

## 5. The outlier problem, and how to fit the ruler

Here is the trap, made concrete. Look at one layer's activation for a token: most values sit near zero, but a few channels spike to 50 or 100 while the rest are under 1. If you pick one scale for the whole row, that scale is set by the spike. Everything small then rounds to almost zero and the signal dies.

![Activations have rare large spikes in a few channels, which break a single uniform scale](/assets/day2-outliers.png)
*Rare spikes in a few channels dominate a single scale and crush the small values.*

This is why "just round the weights to 8 bits" works but "just round the activations to 8 bits" collapses. The weights are calm. The activations are not.

My take: the entire field of quantization for LLMs exists because of a handful of outlier channels. Remember that picture, because every method below is a different answer to it.

## 6. LLM.int8: isolate the outliers

LLM.int8, from Dettmers and others in 2022, takes the simplest possible answer: do not quantize the spikes. Leave the outlier columns in full precision (FP16) and quantize the calm columns to INT8, then add the two results back together.

![LLM.int8 splits the matrix so normal values take the INT8 path and outlier values stay in FP16, then combines](/assets/day2-llm-int8.png)
*Most values use INT8; rare outliers stay in FP16.*

The code is one line:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto",
)
```

![INT4 cuts weight memory and traffic; the compute is not entirely INT4](/assets/day2-int4-inference.png)
*INT4 cuts weight memory and traffic; the compute is not entirely INT4.*

NF4, Normal Float 4, is a 4 bit format tuned to how weights are actually distributed, not a uniform ruler. Its 16 marks sit where weights cluster. It is what QLoRA uses for fine tuning in 4 bit.

My take: LLM.int8 is the honest baseline. It proves the core idea, half the memory with the same quality, using the smallest possible change. Everything after it is about going smaller or faster.

## 7. SmoothQuant, GPTQ, and AWQ

Three methods, three different answers to the same outlier problem.

SmoothQuant attacks outliers from the other side. Every layer is just a multiplication: the input X times the weights W gives the output Y. For any per column scale s, X times W equals (X divided by s) times (W times s). So SmoothQuant can shrink the spike in X and move that size into W instead.

In practice it picks, for each column, a scale s. The formula is s = (biggest activation in that column to the power alpha) divided by (biggest weight in that column to the power 1 minus alpha), with alpha around 0.5.

The symbols: `X` is the activation for that column, `W` is the weight for that column, `s` is the per column scale SmoothQuant computes, and `alpha` is the dial, about 0.5, that splits the difficulty between activations and weights.

The dial alpha decides how much difficulty moves: at 0 all of it stays in X, at 1 all of it moves to W. So X becomes clean INT8 with one scale per token, and W takes on the difference and quantizes well per column. Both sides INT8, no FP16 half, no need to run two separate calculations. SmoothQuant moved the fragile outlier difficulty out of activations and into the calm weights.

![SmoothQuant migrates scaling from activations into weights so the activation distribution becomes easy to quantize to INT8](/assets/day2-smoothquant.png)
*SmoothQuant moves quantization difficulty from activations into weights.*

GPTQ pushes weights to INT4. Its idea: keep the final output of the layer as close as possible, not each weight on its own. It quantizes one column, then immediately fixes the mistake that column would cause in the columns not yet done. By the end every weight is INT4 but the output stays close to the original. The mental model: paint one strip, fix the spill beside it before moving on.

How does it know which weights matter most to the output? From a small sample of normal inputs. How much the output cares about each weight is measured from that sample, using a standard math tool (the Hessian, estimated from X times X over the sample). Weights whose columns matter more get corrected harder. GPTQ also uses group quantization: it shares one scale across every 128 columns. That is the q_group_size you see in the AWQ code below. The symbols: `q_group_size` is how many columns share one scale (128 here), and the Hessian is the stand in for how sensitive the layer output is to each weight, estimated from the activation covariance X transpose X over a small calibration set.

AWQ agrees outliers matter but protects before quantizing. It finds the important columns from the sample inputs: a column is important if its average input size is large, because multiplying a large input by that weight drives the output.

AWQ scales those important columns up through the same X/s times W times s trick, so INT4 spends its 16 marks on what matters, then divides the scale back out at inference. It uses group quantization, one scale per 128 columns, the same q_group_size as GPTQ. No step-by-step fixing. The code is short.

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    safetensors=False,
)
```

![GPTQ and AWQ both reach INT4 weights, but take different paths: GPTQ repairs after, AWQ protects before](/assets/day2-gptq-vs-awq.png)
*GPTQ fixes errors after quantizing; AWQ protects important channels before.*

My take: you do not need to memorize the math. Remember the shape of each answer. SmoothQuant moves difficulty. GPTQ repairs it. AWQ prevents it. All three exist because of the same outlier channels.

## 8. Quantizing the KV cache

Quantizing the KV cache is the same memory idea applied to a different buffer. The KV cache grows every token and fills GPU memory first, so shrinking it fits more sequences.

![FP16 KV cache blocks become smaller INT8 blocks, shrinking per-token memory so more sequences fit](/assets/day2-kv-cache.png)
*A smaller KV cache fits more sequences, but measure the accuracy tradeoff.*

Per token scaling works because KV values stay within a fixed range, and each attention head can use its own scale. Caution: a small error in K or V shifts the attention score, and attention is sensitive, so KV cache stays at INT8 while weights go to INT4. FP8 KV cache is best but needs H100 or A100. On a T4 the KV cache stays FP16 and only the weights are quantized. Shrink what you can.

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

Reproduce it (the code that produced the table above, run on a T4):

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

results = {}
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
    param_mb = sum(p.numel() * p.element_size() for p in m.parameters()) / 1e6
    results[name] = (sum(ts)/3, param_mb, ppl)
    print(f"{name}: {sum(ts)/3:.1f} tok/s | weights {param_mb:.0f} MB | GPU {torch.cuda.max_memory_allocated()/1e6:.0f} MB | ppl {ppl:.2f}")
    del m; torch.cuda.empty_cache()
```

Read it in three passes.

Quality held. Perplexity stays at 24 to 25 across all three. Dropping to half a byte per weight did not break the model. That is the proof the outlier handling works. The tiny row differences are noise from one short sentence, not a trend.

Memory shrank but you cannot see it in the GPU column. Weights fell 250 to 166 to 123 MB, not the clean 125 and 62 you would get if every parameter were 1 or half a byte. Bitsandbytes leaves the embedding and output layer in FP16 and stores scales, so totals are about 166 and 123. The GPU column stays 687 every time because CUDA reserves a few hundred MB of fixed overhead regardless of model size. On a 125M model that overhead hides the saving. On a 70B model the weights are 140 GB and the same halving would show as about 70 GB saved. The win is real, just invisible here.

Speed did the opposite of marketing, and that is the lesson. INT8 dropped to 13.9, slower than FP16. To multiply in INT8 the GPU must dequantize back to FP16 first, an extra step every token. On a 125M model the weights fit in cache, so the bottleneck is that overhead, not bandwidth, and INT8 loses. INT4 quarters the weights, the bandwidth saved beats the dequant cost, and it wins at 55.8. This is the gap from section 1: tok/s equals bandwidth over weight size only when weight reading is the bottleneck. On a toy it is not, so quantization does not speed up decode. On a 70B model it does, for two reasons. First, weight reading dominates, so moving 2x or 4x less data helps. Second, the GPU's Tensor Cores compute INT8 and INT4 matmuls faster than FP16, not just load them faster. Big models win on both bandwidth and compute.

A note on AWQ: the autoawq library does not support OPT-125M, so it has no measured row. It would land near the INT4 row in memory and perplexity, a 4 bit method solving the same outlier problem by protecting salient channels up front.

My take: the number that matters for your career is the memory one. Nobody hires an inference engineer to shave 50 tok/s off a 125M toy. They hire one to fit a 70B model on hardware that should not hold it, or to serve ten times more users on one box. Quantization does that. Speed is a bonus that appears exactly when the model is big enough to need it.

## Takeaway

Quantization is not magic. It is storing the model's numbers with fewer bits.

1. The cost is memory traffic. Decode speed is set by bandwidth over weight size. Shrink the weights and you shrink the cost.
2. Naive quantization fails because activations have rare outlier spikes. A single scale set by the spike crushes the small values.
3. The methods split into three moves. SmoothQuant moves the difficulty into weights, GPTQ repairs it after, AWQ prevents it before. All three exist for the same outlier channels.
4. On a small model quantization saves memory but may not speed things up. On a large model it is the single biggest lever you have.
5. Measure, do not trust. My benchmark held quality at a quarter of the weight size, and the speed win was hidden only because the model was too small for overhead to matter.

That is the whole game of quantization.

## Day 2 summary

![Quantization shrinks weights from FP16 to INT8 to INT4, moving the memory bottleneck and enabling larger models on the same hardware](/assets/day2-quantization-overview.png)
*Quantization trades precision for smaller weights and less memory traffic.*

## Next (Day 3)

The KV cache. Why a long chat eats more GPU memory than the model itself, and how quantizing the cache fixes it.

## Visual references

The diagrams in this article are original illustrations created for this article. Their technical concepts were informed by the following work:

- SmoothQuant (Xiao et al., 2023).
- GPTQ (Frantar et al., 2022).
- AWQ (Lin et al., 2023).
- LLM.int8 (Dettmers et al., 2022).
- NF4 and QLoRA (Dettmers et al., 2023).
- NVIDIA Hopper (FP8) and Tesla T4 datasheets.

## References

- Dettmers et al., LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale, 2022.
- Xiao et al., SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs, 2023.
- Frantar et al., GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers, 2022.
- Lin et al., AWQ: Activation-aware Weight Quantization, 2023.
- Dettmers et al., QLoRA: Efficient Finetuning of Quantized LLMs, 2023.
- NVIDIA Tesla T4 datasheet.
- OPT model card, Meta AI / Hugging Face.
