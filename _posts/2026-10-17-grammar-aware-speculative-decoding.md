---
title: "Speculative Decoding Is Broken Under JSON Schemas — Here's the Fix"
date: 2026-10-17
categories: [llm, inference, optimization, speculative-decoding, vllm]
permalink: /grammar-spec-decoding/
---

<style>
.gsd-fig{margin:28px 0}
.gsd-fig svg{max-width:100%;height:auto;border:1px solid #e6e4e0;border-radius:8px;background:#fff}
.gsd-fig .cap{font-size:.92em;color:#6b6b6b;margin-top:10px;line-height:1.55}
.gsd-chart{border:1px solid #e6e4e0;border-radius:8px;padding:22px 24px 16px;margin:28px 0;background:#fff}
.gsd-chart h4{margin:0 0 18px;font-size:.8em;text-transform:uppercase;letter-spacing:.09em;color:#6b6b6b}
.gsd-bar{display:grid;grid-template-columns:200px 1fr 62px;gap:14px;align-items:center;margin:11px 0}
.gsd-bar .lbl{text-align:right;font-size:.88em;color:#2c2c2c}
.gsd-bar .track{background:#f4f3f1;border:1px solid #e6e4e0;border-radius:5px;height:27px;overflow:hidden}
.gsd-bar .fill{height:100%;display:flex;align-items:center;padding-left:11px;font-size:.78em;color:#fff;font-weight:500}
.gsd-bar .fill.grey{background:#a8a49e}
.gsd-bar .fill.red{background:#b0655a}
.gsd-bar .fill.teal{background:#14544a}
.gsd-bar .val{font-size:.86em;font-family:monospace;color:#1a1a1a}
.gsd-chart .note{font-size:.82em;color:#6b6b6b;margin-top:13px;line-height:1.55}
.gsd-callout{border:1px solid #e6e4e0;border-left:3px solid #14544a;background:#f7f6f4;padding:16px 20px;margin:26px 0;font-size:.97em;line-height:1.65}
.gsd-callout.warn{border-left-color:#9a3524}
.gsd-callout .ck{display:block;font-size:.78em;font-weight:600;letter-spacing:.13em;text-transform:uppercase;color:#14544a;margin-bottom:6px}
.gsd-callout.warn .ck{color:#9a3524}
.gsd-tbl{overflow-x:auto;margin:24px 0}
.gsd-tbl table{border-collapse:collapse;width:100%;font-size:.9em;font-family:-apple-system,sans-serif}
.gsd-tbl th{text-align:left;font-size:.82em;text-transform:uppercase;letter-spacing:.07em;color:#6b6b6b;padding:9px 13px 9px 0;border-bottom:1.5px solid #1a1a1a}
.gsd-tbl td{padding:11px 13px 11px 0;border-bottom:1px solid #e6e4e0;vertical-align:top}
.gsd-tbl tr.win td{background:#f7f6f4}
.gsd-tbl tr.win td:first-child{position:relative;padding-left:13px}
.gsd-tbl tr.win td:first-child::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:#14544a}
.gsd-tbl .mono{font-family:monospace;font-size:.93em}
@media(max-width:640px){.gsd-bar{grid-template-columns:92px 1fr 52px}.gsd-bar .lbl{font-size:.72em}}
</style>

By Yuvraj Singh Bhadoria.

Every AI agent you ship runs on JSON. Tool calls, structured outputs, schema-constrained decoding — that is the workload. And speculative decoding, the standard throughput lever for LLM serving, quietly falls apart on exactly that workload. I measured it, root-caused it, and fixed it. This post covers the whole path — including the three dead ends that shaped the final design.

## 0. The question

Speculative decoding is supposed to be a free win: a small drafter proposes tokens, the big model verifies them in one batched forward pass, and you get the same output faster. Grammar-constrained decoding is also supposed to be a free win: the server guarantees the output matches your JSON schema. Put them together, and one breaks the other. How much does it break, and can it be fixed without giving up either guarantee?

## 1. The problem: drafts can't read the rulebook

Speculative decoding drafts tokens cheaply with a small drafter, then verifies them against the target model in one batched forward pass. Under grammar-constrained decoding, the verifier only accepts grammar-valid tokens — but the drafter knows nothing about the grammar.

<div class="gsd-fig">
<svg viewBox="0 0 720 140" role="img" aria-label="Pipeline: EAGLE3 drafter, grammar mask, target verify">
  <defs><marker id="gsd-ar" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#14544a"/></marker></defs>
  <g font-family="-apple-system,sans-serif" font-size="13.5" text-anchor="middle">
    <rect x="20" y="40" width="160" height="56" rx="6" fill="#fff" stroke="#e6e4e0"/>
    <text x="100" y="64" fill="#1a1a1a" font-weight="600">EAGLE3 head</text>
    <text x="100" y="84" fill="#6b6b6b" font-size="12.5">drafts K tokens</text>
    <line x1="180" y1="68" x2="242" y2="68" stroke="#14544a" stroke-width="1.5" marker-end="url(#gsd-ar)"/>
    <rect x="244" y="40" width="168" height="56" rx="6" fill="#f7f6f4" stroke="#14544a"/>
    <text x="328" y="64" fill="#14544a" font-weight="600">grammar mask</text>
    <text x="328" y="84" fill="#14544a" font-size="12.5">before argmax — the fix</text>
    <line x1="412" y1="68" x2="474" y2="68" stroke="#14544a" stroke-width="1.5" marker-end="url(#gsd-ar)"/>
    <rect x="476" y="40" width="170" height="56" rx="6" fill="#fff" stroke="#e6e4e0"/>
    <text x="561" y="64" fill="#1a1a1a" font-weight="600">target verify</text>
    <text x="561" y="84" fill="#6b6b6b" font-size="12.5">keeps only valid drafts</text>
  </g>
</svg>
<div class="cap">vLLM masks at verification only. The draft step is where 84.9% of the waste happens — that's where this fix goes.</div>
</div>

Measured with vLLM 0.11.2 + xgrammar + Qwen2.5-7B on an A100: under a JSON schema, **84.9% of n-gram drafts were rejected** by the grammar FSM (292 of 344). Unmasked EAGLE3 — the strongest drafter vLLM supports — accepted only **1.44 tokens per propose call** on agentic JSON and ran **0.97× slower than no speculation at all**. Spec decode was not unhelpful here; it was a pessimization.

This is not a niche corner: vLLM's Q4 2026 roadmap carries the open item *"Improve configuration ergonomics and robustness with tool calling, reasoning, and structured outputs."*

## 2. Three dead ends

I kept every negative result — they are what makes the final design make sense.

1. **Truncate invalid drafts.** Rejected — 0.24× vs no-spec. Valid output, but draft length collapses; there is nothing left to speculate on.
2. **Replace invalid drafts with the first bitmask-valid token.** Rejected — valid is not likely. 83.1% of drafts got substituted and were rejected at verification anyway. The verifier wants what the model would actually say.
3. **Run a small standalone drafter with masked logits inside propose().** Rejected — 61× slower (183.5 ms vs 13.5 ms per token). Correct architecture on paper; a second forward pass inside the sampling loop serializes everything.

<div class="gsd-callout"><span class="ck">The lesson</span>The fix has to live inside the shared forward pass, at the point where draft logits already exist — not beside it, and not after the fact.</div>

## 3. The fix: mask the draft's logits, not just the target's

vLLM's EAGLE3 path computes draft logits and immediately argmaxes them. The head's logits are scattered into the *target's* vocabulary space before that argmax — which means a standard xgrammar bitmask applies directly, at every speculative position:

```python
# inside EagleProposer.propose(), per draft position
logits = self.model.compute_logits(hidden, attn)   # target vocab
apply_token_bitmask_inplace(logits, bitmask)       # <- insert here
token = logits.argmax(dim=-1)
```

The mask needs grammar state that advances with each drafted token. The final design builds a **fresh `GrammarMatcher` per propose call**, replays the committed-token history into it, masks all K draft positions, then discards it — zero `rollback()` calls. Why not reuse one matcher and roll back the drafted-but-rejected window: xgrammar 0.1.25's `rollback()` desynced after a permanent accepted prefix when the window held forced-continuation tokens (for example char-splatted enum values). The matcher returns without error but stops reflecting the true grammar state.

If the draft-vocab subset contains no grammar-valid token for a row, the fallback takes the first grammar-valid token. Measured fallback rate: **0 of 4,486 masks**.

## 4. Results

Weather-tool JSON schema — Qwen2.5-7B-Instruct plus its EAGLE3 head, A100, batch=1, K=5, 12 distinct prompts, 3 timing iterations with rotated config order, diagnostics gated out of the hot path. Lower TPOT (time per output token) is better:

<div class="gsd-chart">
  <h4>Time per output token (ms) — lower is better</h4>
  <div class="gsd-bar"><div class="lbl">no speculation</div><div class="track"><div class="fill grey" style="width:88.7%">11.09</div></div><div class="val">1.00×</div></div>
  <div class="gsd-bar"><div class="lbl">EAGLE3 (unmasked)</div><div class="track"><div class="fill red" style="width:91.7%">11.47</div></div><div class="val">0.97×</div></div>
  <div class="gsd-bar"><div class="lbl"><strong>EAGLE3 + grammar mask</strong></div><div class="track"><div class="fill teal" style="width:76.8%">9.61</div></div><div class="val">1.15×</div></div>
  <div class="note">vs no-spec. The red bar is slower than doing nothing. Std over 3 iterations: 0.01 / 0.04 / 0.03 ms.</div>
</div>

<div class="gsd-tbl">
<table>
  <thead><tr><th>Config</th><th>Valid JSON</th><th>TPOT (ms)</th><th>tok/propose</th><th>vs baseline</th></tr></thead>
  <tbody>
    <tr><td>no_spec</td><td>33/36</td><td class="mono">11.09 ± 0.01</td><td class="mono">—</td><td>—</td></tr>
    <tr><td>baseline EAGLE3 (unmasked)</td><td>33/36</td><td class="mono">11.47 ± 0.04</td><td class="mono">1.443</td><td>1.00×</td></tr>
    <tr class="win"><td><strong>EAGLE3 + grammar mask</strong></td><td>33/36</td><td class="mono"><strong>9.61 ± 0.03</strong></td><td class="mono"><strong>1.969 (+36.5%)</strong></td><td class="mono"><strong>1.193×</strong></td></tr>
  </tbody>
</table>
</div>

Acceptance rises because **39.1% of draft-position argmaxes change** under the mask (1,756 of 4,486 positions): the mask removes grammar-invalid high-probability drafts, so the next-most-likely *valid* token is drafted instead. The advantage holds at every draft length tested — K=3: 1.15×, K=5: 1.19×, K=7: 1.17× (best absolute: **K=3 + mask = 1.25× vs no-spec**).

### The harder the schema, the bigger the win

On a second, enum-heavy tool-call schema (nested enums and arrays), the story gets stronger — including a finding I did not expect:

<div class="gsd-tbl">
<table>
  <thead><tr><th>Config</th><th>Valid JSON</th><th>TPOT (ms)</th><th>tok/propose</th></tr></thead>
  <tbody>
    <tr><td>no_spec</td><td>5/8</td><td class="mono">11.02</td><td class="mono">—</td></tr>
    <tr><td>baseline EAGLE3 (unmasked)</td><td class="mono" style="color:#9a3524">3/8</td><td class="mono">13.54</td><td class="mono">1.203</td></tr>
    <tr class="win"><td><strong>EAGLE3 + grammar mask</strong></td><td class="mono"><strong>5/8</strong></td><td class="mono"><strong>10.37 (1.31× vs baseline)</strong></td><td class="mono"><strong>1.807 (+50.2%)</strong></td></tr>
  </tbody>
</table>
</div>

<div class="gsd-callout warn"><span class="ck">Unexpected finding</span>Constrained spec decode <strong>degrades output validity</strong> on hard schemas — unmasked EAGLE3 produced 3/8 valid outputs vs 5/8 for no-spec, pushed into degeneration loops until the token cap. Grammar-masking the drafts restores validity to the no-spec level (5/8) while also being the fastest configuration. And the acceptance boost is <em>larger</em> on the schema with more enums (+50.2% vs +36.5%) — enums are exactly where unconstrained drafts wander off.</div>

## 5. Takeaways

> Grammar-valid is not model-likely — and model-likely is not grammar-valid. Under constrained sampling, speculative decoding needs both, at the draft step, where it is cheap.

1. The problem is real and large: **84.9% of n-gram drafts are grammar-invalid** under a JSON schema.
2. **Grammar-valid is not model-likely:** first-valid-token substitution gets rejected anyway (83.1% substituted, still rejected).
3. **A drafter must share the target's forward pass:** a standalone drafter inside `propose()` is 61× slower.
4. **Masking after the d2t scatter flips the economics:** a net-negative speculative config (0.967× vs no-spec) becomes net-positive (1.154×).
5. **Harder schema, larger acceptance win** (+36.5% → +50.2%) — enums are where drafts wander off.
6. **Constrained spec decode can degrade validity;** grammar-masking the drafts restores it.

The prototype is a monkey-patched `EagleProposer` — the shape of a real vLLM change is clear: a bitmask hook in the draft loop plus a per-request matcher lifecycle. I have filed it upstream as a [design issue on vLLM](https://github.com/vllm-project/vllm/issues/62130). Open questions I would like to push on: batch > 1, a GPU-resident FSM to remove the remaining per-position host sync, and whether the same idea transfers to MTP heads.

**Links:** [Repository — grammar-spec-decode](https://github.com/YuvrajSinghBhadoria2/grammar-spec-decode) · [vLLM issue #62130](https://github.com/vllm-project/vllm/issues/62130)

---

*Method note: single-session measurements — batch=1, greedy decoding, one A100, one target/head pair (Qwen2.5-7B plus a community EAGLE3 head), two JSON schema classes, 12 prompts (weather) / 8 prompts (tool schema), 3 timing iterations with rotated order, diagnostics off in the hot path. Timing numbers were independently re-derived from the raw JSONs post-session — all confirmed. The `rollback()` unsoundness was observed in-session (xgrammar 0.1.25); a minimal repro is available on request. Limitations and all negative results are documented in [RESULTS.md](https://github.com/YuvrajSinghBhadoria2/grammar-spec-decode).*
