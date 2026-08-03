# Fused residual + RMSNorm

Contract:

```text
y = RMSNorm(x + residual; weight)
x, residual: contiguous [rows, hidden]
weight: contiguous [hidden]
accumulation: FP32
```

The dispatcher uses the Triton path only after the exact GPU/dtype/shape key passes the correctness sweep. Unsupported, unvalidated, or failed shapes use the PyTorch reference. Microbenchmark results alone must not be described as an end-to-end LLM speedup.
