# 14 · What the Finished Product Should Look Like

## 14.1 使用者體驗

### Step 1 — Inspect

```bash
edgeflow inspect
```

輸出：

```text
GPU       RTX 4080 SUPER · 16 GB · SM 8.9
Runtime   CUDA / PyTorch / Triton
Backends  PyTorch ✓  llama.cpp ✓  vLLM ✓
Profiler  torch.profiler ✓  Nsight Systems ✓  Nsight Compute ✓/permission
Status    Ready for validated runs
```

### Step 2 — Define workload

```bash
edgeflow workload create \
  --model ministral3-3b \
  --profile local-agent \
  --prompt-distribution 512:0.25,1024:0.45,2048:0.30 \
  --output 128 \
  --concurrency 1 \
  --session-requests 20
```

### Step 3 — Screen

```bash
edgeflow tune screen --workload local-agent.json
```

顯示candidate、預估時間與quality requirements。

### Step 4 — Diagnose

```bash
edgeflow profile --run <run_id> --level systems
edgeflow diagnose --run <run_id>
```

輸出hypothesis與下一個controlled experiment，不直接宣稱原因。

### Step 5 — Verify

```bash
edgeflow experiment run <experiment_id>
edgeflow validate <run_pair_id>
```

### Step 6 — Policy

```bash
edgeflow policy build --workload local-agent.json
edgeflow policy show <policy_id>
```

輸出：

```text
Short session / C=1 / prompt≤1024 → llama.cpp Q6
Persistent / C=1 / prompt≤1024  → torch.compile reduce-overhead BF16
Concurrency≥4                   → vLLM plan v7
Fallback                        → PyTorch eager BF16
```

每條旁邊都有quality、CI、break-even與evidence ID。

---

## 14.2 Dashboard

首頁：

- RTX 4080 SUPER狀態；
- 目前validated policy；
- workload卡；
- cold vs steady選擇；
- top bottleneck evidence；
- quality Pareto；
- run health。

使用者點policy rule後，看到：

```text
Measured runs
→ Comparison
→ Profiler observation
→ Intervention
→ Validation
→ Rule
```

---

## 14.3 GitHub Demo Script（90秒）

1. 開README，讀一句定位。
2. 看dashboard screenshot。
3. 執行 `edgeflow inspect`。
4. 載入已附的demo dataset，不需模型權重。
5. 在UI選 local-agent workload。
6. 顯示為何cold winner與steady winner不同。
7. 點一個evidence chain。
8. 打開custom kernel heatmap與end-to-end結果。
9. 執行 `python scripts/verify_results.py`。

---

## 14.4 最終 README Headline（數字完成後填）

```text
On an RTX 4080 SUPER, EdgeFlow reduced expected session cost by X%
versus the best fixed validated plan across the measured workload distribution,
while satisfying the selected quality constraint.
```

第二claim：

```text
Profiler-guided interventions reached ε-best performance in Y fewer trials
than grid/random/TPE under the same candidate space and trial accounting.
```

第三claim只有E24通過：

```text
The validated Triton path reduced <operator> time by X% and end-to-end TPOT by Y%
in its dispatch region, with automatic fallback elsewhere.
```

不得在沒有實驗前填入示意數字。

---

## 14.5 履歷成果

### ML Systems版本

- Built an evidence-backed local LLM inference autotuner across PyTorch, `torch.compile`, llama.cpp, and vLLM on an RTX 4080 SUPER.
- Designed workload-conditioned policies that account for TTFT, TPOT, tail latency, VRAM, quality, and compile/load amortization.
- Implemented profiler-guided controlled interventions with traceable evidence chains and correctness-gated recommendations.
- Implemented and integrated a Triton optimization with full shape-sweep validation and runtime fallback. *(only after E24)*

### Research版本

- Formulated local LLM deployment as constrained policy selection over workload distributions rather than single-configuration benchmarking.
- Evaluated causal search efficiency, amortization-aware objectives, policy regret, and quality-constrained Pareto frontiers.

---

## 14.6 未來延伸

完成EdgeFlow核心後，才加入：

1. FAD-aware shared weights／adaptive exit runtime；
2. cross-GPU transfer；
3. RTX 30/40/50 series policy transfer；
4. TensorRT-LLM backend；
5. speculative decoding；
6. energy-aware objective；
7. learned cost model；
8. grounded Performance Copilot。

FAD整合會是自然extension，但不應阻礙EdgeFlow本身成為完整作品。
