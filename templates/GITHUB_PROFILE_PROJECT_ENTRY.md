## EdgeFlow · Local LLM Performance Scientist

**Causal, workload-conditioned autotuning for local LLM inference on RTX GPUs.**

- Cross-runtime measurement: PyTorch, `torch.compile`, llama.cpp, and vLLM.
- Profiler-grounded interventions rather than black-box winner picking.
- Session-aware policy includes load, compile, graph capture, memory, latency, and quality.
- Correctness-gated Triton optimization with full shape sweep and fallback dispatch.

**Evidence to show:** one policy-vs-fixed chart, one causal evidence chain, one kernel speedup region, and one reproducibility badge. Avoid a wall of badges or unvalidated headline claims.
