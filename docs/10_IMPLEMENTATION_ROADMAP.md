# 10 · Implementation Roadmap

## 10.1 原則

每一 phase 都要有可展示產物與 exit criteria；不能等全部完成才有結果。

---

# Phase 0 · Repository and Environment

## Deliverables

- repo scaffold；
- license（建議 Apache-2.0）；
- `uv`／lockfile；
- WSL2 setup；
- hardware inspect；
- schema；
- CI；
- smoke model。

## Tasks

1. 建立 `edgeflow` package與CLI。
2. 實作hardware fingerprint。
3. 定義ModelSpec／WorkloadSpec／ExecutionPlan／RunManifest。
4. 建立SQLite migration。
5. PyTorch CUDA與Triton smoke。
6. 加入secret scan與schema test。

## Exit criteria

```bash
edgeflow inspect --json
edgeflow doctor
python -m pytest
```

全部pass，fingerprint可保存。

---

# Phase 1 · Trustworthy Measurement

## Deliverables

- PyTorch eager adapter；
- workload generator；
- token timestamp；
- metrics；
- validation v0；
- E01–E04結果。

## Tasks

- engine-only timing；
- e2e local HTTP timing；
- warmup convergence；
- thermal monitor；
- raw JSONL；
- HTML report v0。

## Exit criteria

- 重跑同configuration變異在門檻內；
- timer calibration完成；
- raw data可重建summary；
- invalid background load能被拒絕。

---

# Phase 2 · Runtime Adapters

## Deliverables

- `torch.compile` adapter；
- llama.cpp adapter；
- vLLM adapter；
- fairness audit；
- primary model registry。

## Tasks

- process lifecycle；
- server readiness；
- tokenizer parity；
- load/compile拆分；
- native metric parser；
- backend capability probe。

## Exit criteria

- E05–E09完成；
- 至少一個模型可跨三runtime公平比較；
- unsupported capability不會crash整個matrix。

---

# Phase 3 · Policy and Causal Loop

## Deliverables

- candidate generator；
- deterministic optimizer；
- bottleneck rules；
- evidence graph；
- policy JSON；
- E10–E20。

## Tasks

- grid/random/TPE baselines；
- Pareto；
- session objective；
- profiler adapters；
- matched experiment builder；
- policy scope與fallback。

## Exit criteria

- 可生成至少兩條不同workload rule；
- 三個hypothesis有supported/rejected evidence；
- policy在holdout replay可評估；
- cold與steady winner可比較。

---

# Phase 4 · Custom Triton Optimization

## Deliverables

- kernel candidate report；
- Triton kernel；
- correctness suite；
- heatmap；
- dispatch；
- end-to-end integration。

## Exit criteria

- E21–E24完成；
- 全shape結果公開；
- fallback可靠；
- headline只在end-to-end通過時使用。

---

# Phase 5 · Local-first Web Console and Portfolio Release

## Deliverables

- localhost-only Web UI；
- typed single-GPU worker queue；
- optional read-only public export；
- README；
- results／audit；
- demo GIF；
- reproducibility bundle；
- v0.1 release。

## Exit criteria

- UI所有數字可追raw run；
- 不含fake result；
- 非 loopback bind、CSRF、cross-origin write、任意 shell/path 必須被拒絕；
- worker failure／cancel 不污染其他 run；
- clean machine smoke成功；
- 90秒demo可完整執行；
- release checklist全部完成。

---

# Phase 6 · Optional Learned Components

僅在run DB足夠後：

- cost model；
- bottleneck classifier；
- pruning ablation；
- model card。

若資料不足，保持deterministic，不為了AI標籤勉強訓練。

---

# Phase 7 · Optional Copilot

- tool server；
- local/API model；
- grounding eval；
- UI evidence drawer；
- no arbitrary shell。

只有E28通過才公開。

---

## 10.2 建議 work breakdown

| Workstream | 主要檔案 | 驗收 |
|---|---|---|
| Core schemas | `specs/` | schema tests |
| Hardware | `edgeflow/hardware` | fingerprint |
| Runtime | `edgeflow/runtimes` | adapter conformance |
| Metrics | `edgeflow/metrics` | timer calibration |
| Validation | `edgeflow/validation` | verdict fixtures |
| Profiler | `edgeflow/profiler` | parsed summaries |
| Causal | `edgeflow/diagnosis` | intervention records |
| Optimizer | `edgeflow/optimizer` | baseline comparison |
| Kernel | `edgeflow/kernels` | correctness + heatmap |
| UI | `dashboard/` | usability tasks |
| Docs | `docs/` | audit completeness |

---

## 10.3 不可提前做的事

- 在timer還不可信前跑大matrix。
- 在fairness audit前宣稱跨runtime winner。
- 在profile前選custom kernel。
- 在quality gate前推薦Q4。
- 在run DB很少時訓練cost model。
- 在evidence schema不穩定時做Copilot。
- 在有demo數字時把網站公開成results頁。

---

## 10.4 Milestone 命名

- `M0 Trust the Clock`
- `M1 Compare Fairly`
- `M2 Explain the Bottleneck`
- `M3 Select by Workload`
- `M4 Optimize the Hot Path`
- `M5 Publish the Evidence`
- `M6 Learn from Runs`（optional）
- `M7 Ask the Copilot`（optional）
