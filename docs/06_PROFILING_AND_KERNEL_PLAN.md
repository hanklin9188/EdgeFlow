# 06 · Profiling and Custom Kernel Plan

## 6.1 原則

Custom kernel 不是為了在 README 放上「我會 Triton」，而是要完成：

```text
Profile → Identify high-impact opportunity → Implement → Validate → Dispatch → Confirm end-to-end
```

如果 profile 顯示某 operator 只佔 0.5%，即使 microbenchmark 快 2×，也不應是第一個 kernel。

---

## 6.2 Profiling 層級

### L0 — Production timing

工具：

- `time.perf_counter_ns`；
- CUDA Events；
- backend token timestamps；
- `nvidia-smi` sampling。

低干擾，正式 latency用此層。

### L1 — PyTorch Profiler

收集：

- CPU/CUDA activities；
- operator names；
- input shapes（只在diagnostic）；
- memory；
- stacks（必要時）；
- Torch-Compiled Region；
- graph breaks／recompile IDs；
- Triton kernel names。

注意：`record_shapes`、stack、memory profiling有額外 overhead；不拿來報正式性能。

### L2 — Nsight Systems

收集：

- CUDA runtime/driver calls；
- kernel timeline；
- memcpy；
- CPU scheduling；
- NVTX ranges；
- CUDA Graph trace；
- cuBLAS trace（必要時）。

EdgeFlow 應在關鍵區段加 NVTX：

```text
edgeflow::request
edgeflow::tokenize
edgeflow::prefill
edgeflow::decode_step
edgeflow::sampling
edgeflow::serialize
```

Trace parser 輸出：

- kernel count/token；
- median kernel duration；
- gap ratio；
- CPU launch latency；
- H2D/D2H bytes；
- prefill/decode time share。

### L3 — Nsight Compute

只對 top kernel／custom kernel；收集最小必要 metric set，避免 replay overhead過大。

關注：

- achieved occupancy；
- SM throughput；
- tensor activity；
- DRAM throughput；
- L2 hit；
- warp stalls；
- register/shared memory；
- roofline相關指標。

在 WSL2 上需確認 performance counter permission；權限不足時標記 `profile_unavailable`，不能猜測。

---

## 6.3 Profiler Overhead Calibration

對同一 workload：

1. unprofiled正式 run；
2. L1；
3. L2；
4. L3 target kernel。

報 overhead ratio，但不嘗試用簡單比例修正 profiler latency。Profiler trace只用于結構判斷。

---

## 6.4 第一個 Kernel 的選擇機制

Phase 3 完成前，保留三個候選。

### Candidate A — Fused residual + RMSNorm

形式：

\[
y=\operatorname{RMSNorm}(x+r;\gamma).
\]

機會：減少中間 tensor與kernel launch。

風險：現有 compiler／runtime可能已融合；需先檢查生成 graph。

### Candidate B — Fused gated activation epilogue

Llama-like FFN中：

\[
z=\operatorname{SiLU}(g)\odot u,
\]

若 gate/up projection輸出已存在，可融合 activation、mul與後續 residual-related epilogue。

機會：memory-bound pointwise、頻率高。

風險：大型 GEMM仍主導，端到端gain可能小。

### Candidate C — Small-rank projection + fused epilogue

即使尚未整合 FAD，也可做通用 PEFT／adapter module：

\[
y=r+s+\alpha B(Ax).
\]

第一版可融合第二個 projection epilogue、scale、two adds。

機會：batch-1小rank、launch overhead；與未來 FAD自然銜接。

風險：EdgeFlow目前主線不依賴adapter，需用獨立 plugin呈現，不能讓作品看似只為FAD。

### 選擇分數

\[
Score = T_{share}\times F_{launch}\times O_{fusion}\times C_{feasible}\times G_{portfolio}.
\]

- `T_share`：time share；
- `F_launch`：每token頻率；
- `O_fusion`：可省memory/launch；
- `C_feasible`：在時程內；
- `G_portfolio`：能清楚展示systems reasoning。

---

## 6.5 Triton 實作規範

每個 kernel 需要：

```text
reference.py
kernel.py
autotune.py
dispatch.py
test_correctness.py
bench_micro.py
bench_integration.py
README.md
```

### Reference

- 純 PyTorch；
- 禁止就地修改輸入，除非 kernel contract就是in-place；
- 定義 shape/dtype/stride contract。

### Kernel

- explicit mask；
- 非2次方 shape；
- FP32 accumulation規則；
- numerical behavior文件化；
- 避免依賴未公開 internal API，或清楚 pin version。

### Autotune

Autotune key只放真正影響最佳 config 的 shape特徵，避免 cache explosion。

保存：

- selected config；
- compile time；
- tune time；
- cache key；
- GPU fingerprint。

### Dispatch

```python
if not capability_ok:
    return reference(...)
if shape in known_bad_region:
    return reference(...)
if validation_cache.get(kernel_hash, shape, dtype) != "PASS":
    return reference(...)
return triton_kernel(...)
```

永遠提供 fallback。

---

## 6.6 Correctness Sweep

### Shapes

- token rows：1、2、4、8、16、32、128、512；
- hidden：768、1024、1536、2048、3072、4096；
- intermediate依模型；
- rank（若adapter）：8、16、32、64、128；
- awkward：1000、3073、4095。

### Dtypes

- FP32 development；
- FP16；
- BF16。

### Inputs

- normal random；
- uniform；
- large magnitude；
- near-zero；
- mixed signs；
- NaN/Inf negative tests；
- non-contiguous或明確拒絕。

### Threshold

預設：

| dtype | atol | rtol |
|---|---:|---:|
| FP32 | 1e-4 | 1e-4 |
| FP16 | 2e-2 | 2e-2 |
| BF16 | 5e-2 | 5e-2 |

實際門檻需依運算深度調整並在 test file固定；不能為了讓錯誤 kernel 通過而事後放寬。

另報：

- max abs；
- max rel（排除near-zero）；
- cosine；
- NaN/Inf；
- deterministic repeat。

---

## 6.7 Microbenchmark Protocol

- 每 shape先 25 warmup；
- 100–500 measured iterations；
- CUDA Events；
- 每 shape randomize implementation order；
- 重啟 process抽查cache effect；
- PyTorch eager、`torch.compile` equivalent、Triton三者；
- median與p95；
- 報 compile/autotune separately；
- 速度圖顯示全部shape，不隱藏slow region。

計算：

\[
Speedup=\frac{T_{baseline}}{T_{triton}}.
\]

另估 effective bandwidth／FLOPs，但清楚標示是估計。

---

## 6.8 Integration Protocol

1. 在模型 module中以 feature flag替換。
2. 先比較 layer output。
3. 再比較 logits／tokens。
4. 先跑短 smoke，再完整 benchmark。
5. 量測 launch count變化。
6. 量測 top kernel time share。
7. unprofiled正式 latency。
8. quality gate。

若 micro speedup明顯、end-to-end neutral，可能原因：

- operator占比小；
- compiler本來已融合；
- launch savings被其他同步抵銷；
- dispatch overhead；
- shape頻率不同；
-模型主 bottleneck在GEMM/attention。

這仍是有價值結果，但不能標「LLM inference speedup」。

---

## 6.9 Kernel Result Page

GitHub頁面應包含：

- motivation trace截圖；
- operator equation；
- reference vs kernel；
- contract；
- correctness table；
- speedup heatmap；
- slower region；
- dispatch rule；
- end-to-end result；
- limitations；
- exact hardware/software。

Headline不要只寫 `up to 2.3×`；至少並列 geometric mean或weighted mean與end-to-end。
