---
title: "The KV Cache for LLM Inference: Why Long Contexts Blow Up Memory, and a Measured Size Breakdown"
date: 2026-08-18
categories: [llm, inference, optimization, kv-cache]
permalink: /day-3/
---

By Yuvraj Singh Bhadoria.

The last post was about quantization, shrinking the weights so the model takes less space. This post goes after the other thing that eats GPU memory: the KV cache. The weights are fixed, but the cache grows every time you send a word. I am writing these as I learn, working the numbers out instead of trusting marketing.

## 0. The question

When you chat with a model, why does a long conversation get slower and eventually run out of memory? The weights never change. So what is actually filling the GPU? The answer is the KV cache, and almost nobody counts it until it bites them.

## 1. What the KV cache is

Plain version: to write the next word, the model needs to pay attention to every word you have sent before. Instead of recomputing that attention from scratch each time, it saves two small vectors for each past word, in every layer, for every attention head. Those vectors are called Key and Value. The saved memory is the KV cache. It is the model's running memory of the conversation.

![KV cache basics: each input token produces a Key and Value vector that accumulate in GPU memory, one K/V pair per layer per attention head, growing with the conversation](/assets/day3-kv-cache-basics.png)
*The cache is the model's running memory of the conversation — one K/V pair stored per token, per layer, per head.*

The cost is simple: one token in, a fixed block of memory out, every layer, every head, both Key and Value. So the cache grows in lockstep with the conversation.

## 2. How big is it (worked out from the model shape)

For OPT-125M the shape is 12 layers, 12 heads, and 64 numbers per head. The KV cache for a single token is 2 (Key and Value) times 12 layers times 12 heads times 64 numbers. At fp16 that is about 0.035 MB per token. Multiply by the conversation length and you get the total.

| Context | KV fp16 | KV int8 |
|---|---|---|
| 512 tokens | 18.9 MB | 9.4 MB |
| 2048 tokens | 75.5 MB | 37.7 MB |
| 4096 tokens | 151.0 MB | 75.5 MB |
| 8192 tokens | 302.0 MB | 151.0 MB |

![KV cache size growth: a stacked bar per context length (512 / 2048 / 4096 / 8192) showing the cache growing linearly, fp16 vs int8](/assets/day3-kv-size-growth.png)
*Cache size grows linearly with context; int8 is exactly half of fp16 at every length.*

I worked these out from the model's exact dimensions, then confirmed them on a T4 by reading the actual cache tensors the model built. The run matched the formula exactly up to OPT-125M's 2048-token limit (75.5 MB at 2048). The 8192-token row is the formula carried forward, since this small model cannot be run that long.

The surprise: at 8192 tokens the KV cache (302 MB) is larger than the entire 250 MB model. On a long chat, the cache, not the weights, is what fills the GPU.

![KV cache vs model weights: the 302 MB KV cache bar towering over the 250 MB model bar at 8192 tokens](/assets/day3-kv-vs-model.png)
*At long context the cache alone is bigger than the whole model.*

My take: most people optimizing inference only think about the weights, which was the whole point of Day 2. The KV cache is the silent second bill. On a 70B model at 32k context it becomes enormous, and it is what actually caps how many users you can serve at once.

## 3. Why it limits you

GPU memory is fixed. The weights take their slice. The rest is shared by the KV caches of every active user. A bigger cache per user means fewer users fit. So the KV cache, not the weights, is the real limit on concurrency.

![Concurrency limit: a fixed GPU memory bar split into the model weights plus one KV-cache slot per active user, showing fewer users fit as the per-user cache grows](/assets/day3-concurrency-limit.png)
*Fixed GPU memory: weights plus one cache per user. Bigger cache per user = fewer users.*

## 4. Shrink it like the weights

Same trick as Day 2, applied to the cache: store the Key and Value in fewer bits. int8 halves the size, which is the right column in the table above. fp8 is best on H100 or A100 class GPUs. On a T4 the cache stays fp16 because the hardware is old, but the math is identical.

![KV cache quantization: an fp16 K/V block next to its smaller int8 version, showing half the per-token memory so more sequences fit](/assets/day3-kv-quantization.png)
*Storing K and V in int8 halves the per-token cache, so more sequences fit on the same GPU.*

The caution from Day 2 still applies. A small error in K or V shifts the attention score, and attention is sensitive, so the KV cache stays at int8 while weights can go to int4. Measure the quality hit on your own prompts before you depend on it.

My take: quantize the KV cache only after you have measured the quality hit. Attention is unforgiving, so earn the 2x memory win before you trust it in production.

## 5. The measured sizes

The table in section 2 is the KV cache size for OPT-125M at four context lengths, worked out from the real model shape. The trend is the point: linear growth with context, and a clean 2x saving from int8.

## Takeaway

1. The KV cache is the model's memory of the conversation. It grows with every token.
2. At long context it can be larger than the weights (302 MB vs 250 MB at 8192 tokens).
3. It, not the weights, is what limits how many users fit on one GPU.
4. Quantizing it to int8 halves the size, the same trick as Day 2, with a quality caveat.
5. Measure, do not trust. The cache is the silent memory bill.

## Reproduce it

All the numbers above come from a notebook I ran on a T4 with plain Hugging Face transformers. The full notebook is in the repo as `day-03-kv-cache.ipynb`. Load the model first, then run these three cells.

**Cell 1 — size from the model shape:**

```python
# KV cache size from the model shape. 2 = Key and Value.
cfg = model.config
n_layers = cfg.num_hidden_layers
n_heads = cfg.num_attention_heads
head_dim = cfg.hidden_size // n_heads

def kv_mb(seq_len, dtype_bytes=2):
    return 2 * n_layers * n_heads * head_dim * seq_len * dtype_bytes / 1e6

print(f"layers={n_layers} heads={n_heads} head_dim={head_dim}")
for s in [256, 512, 1024, 2048]:
    fp16 = kv_mb(s)
    int8 = kv_mb(s, 1)
    print(f"seq {s}: KV fp16 {fp16:.1f} MB | int8 {int8:.1f} MB (2x less)")
```

**Cell 2 — read the actual cache the model stores:**

```python
# Direct KV cache measurement: read the actual cache tensors the model stores.
def kv_cache_mb(seq_len, dtype_bytes=2):
    torch.cuda.empty_cache()
    tok_id = int(tok("a", return_tensors="pt").input_ids[0, 0])
    ids = torch.full((1, seq_len), tok_id, dtype=torch.long, device="cuda")
    with torch.no_grad():
        out = model(ids, use_cache=True)
    pkv = out.past_key_values
    total = 0
    if hasattr(pkv, "key_cache"):                 # DynamicCache (newer transformers)
        for t in pkv.key_cache + pkv.value_cache:
            total += t.numel() * dtype_bytes
    else:                                         # tuple of (key, value) per layer
        for kv in pkv:
            for t in kv:
                if t is not None:
                    total += t.numel() * dtype_bytes
    return total / 1e6

max_len = model.config.max_position_embeddings
print(f"model max context = {max_len} tokens")
for s in [256, 512, 1024, 2048]:
    print(f"seq {s}: KV cache {kv_cache_mb(s):.1f} MB | int8 {kv_cache_mb(s, 1):.1f} MB")
```

**Cell 3 — measure the fp16 cache vs the int8 copy (the saving):**

```python
# Real KV cache quantization: measure the cache memory the GPU actually uses
# in fp16 (what the model builds) and in int8 (quantized copies).
def kv_cache_fp16(seq_len):
    torch.cuda.empty_cache()
    tok_id = int(tok("a", return_tensors="pt").input_ids[0, 0])
    ids = torch.full((1, seq_len), tok_id, dtype=torch.long, device="cuda")
    before = torch.cuda.memory_allocated()
    with torch.no_grad():
        out = model(ids, use_cache=True)
    pkv = out.past_key_values   # keep only the cache alive
    del out                     # free the large output logits tensor
    after = torch.cuda.memory_allocated()
    del pkv
    return (after - before) / 1e6

def kv_cache_int8(seq_len):
    torch.cuda.empty_cache()
    tok_id = int(tok("a", return_tensors="pt").input_ids[0, 0])
    ids = torch.full((1, seq_len), tok_id, dtype=torch.long, device="cuda")
    with torch.no_grad():
        out = model(ids, use_cache=True)
    pkv = out.past_key_values
    if hasattr(pkv, "key_cache"):
        caches = pkv.key_cache + pkv.value_cache
    else:
        caches = [t for kv in pkv for t in kv if t is not None]
    del out
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    stored = []
    for t in caches:
        scale = t.abs().max() / 127 + 1e-8
        stored.append((t / scale).round().clamp(-127, 127).to(torch.int8))
    after = torch.cuda.memory_allocated()
    del stored
    return (after - before) / 1e6

for s in [512, 1024, 2048]:
    fp16 = kv_cache_fp16(s)
    int8 = kv_cache_int8(s)
    print(f"seq {s}: fp16 {fp16:.1f} MB | int8 {int8:.1f} MB | saved {fp16 - int8:.1f} MB ({(1 - int8/fp16)*100:.0f}% less)")
```

## Day 3 summary

![KV cache basics: the model computes Key and Value for each token and feeds them to the attention mechanism, accumulating as the cache](/assets/day3-summary-attention.png)
*Every token produces Key and Value vectors that the attention mechanism caches — the cache is the model's memory of the conversation.*

![Paged attention: the KV cache split into fixed-size blocks shared across sequences in GPU VRAM](/assets/day3-summary-paged.png)
*Serving systems like vLLM manage this growing cache with paged attention, packing many sequences into one GPU. More on that in a later post.*

## Next (Day 4)

The rarest skill in this whole field: writing the GPU kernels that make all of this fast. A hands-on Triton kernel, measured.

## Visual references

The diagrams in this article are original illustrations created for this article. Their technical concepts were informed by:

- vLLM documentation, KV cache management and prefix caching.
- Dao et al., FlashAttention: Fast and Memory-Efficient Attention, 2022.
- NVIDIA FP8 and Tesla T4 datasheets.

## References

- vLLM documentation, KV cache management and prefix caching.
- Dao et al., FlashAttention: Fast and Memory-Efficient Attention, 2022.
- NVIDIA FP8 and Tesla T4 datasheets.
- OPT model card, Meta AI and Hugging Face.
