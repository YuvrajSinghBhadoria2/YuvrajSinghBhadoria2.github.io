---
title: "LLM Inference Optimization — Start Here"
date: 2026-08-15
categories: [llm, inference, optimization]
permalink: /start-here/
---

This is a hands-on series about making large language model inference cheaper and faster, written as I learn it. Every technique is measured on real hardware, not taken from marketing slides.

Who it is for: engineers who want to understand ML systems and GPU inference internals, students breaking into AI infrastructure, and anyone evaluating inference-serving tools.

The plan:

- Day 1: Why inference is expensive (prefill, decode, KV-cache, PagedAttention) with a measured baseline.
- Day 2+: Quantization, batching schedulers, speculative decoding, and more, each re-measured against the Day 1 number.

Start here: [Day 1 — Why LLM Inference Is Expensive](/day-1/)
