# Implementation Status

更新日期：2026-08-04。此表把「工程能力完成」與「需要長時間正式實驗才能成立的研究結論」分開，避免以程式存在冒充實驗完成。

| Phase | Engineering | Validation on this checkout | Public claim status |
|---|---|---|---|
| M0 Trust the Clock | 完成 fingerprint、doctor、raw timing、warmup split、artifact hash | unit/integration pass；RTX 4080 SUPER 可用 | 無性能 headline |
| M1 Compare Fairly | 四 adapter contract；PyTorch 可執行；llama.cpp/vLLM capability-safe | PyTorch smoke 可跑；外部 runtimes 未安裝時 SKIPPED | 跨 runtime 結論未建立 |
| M2 Explain Bottleneck | profiler schema、Nsight command、deterministic diagnosis | rule tests pass | diagnosis 僅 HYPOTHESIS |
| M3 Select by Workload | candidate pruning、session objective、policy/fallback/drift | policy tests pass | 需 formal eligible rows + holdout |
| M4 Optimize Hot Path | fused residual+RMSNorm reference/Triton/cache/fallback | GPU correctness command可執行 | end-to-end claim 未建立 |
| M5 Publish Evidence | CLI/API/dashboard/SQLite/verification/CI | production UI 無 mock 數字 | release 可展示工程，不宣稱未量測加速 |
| M6 Learned component | intentionally deferred | run DB 未達 2,000 unique points | 不訓練 |
| M7 Copilot | contract保留，核心不依賴 Agent | grounding dataset 尚未建立 | 不公開 run-specific回答 |

## Exit criteria mapping

- `edgeflow inspect --json`：已完成。
- `edgeflow doctor`：已完成，optional backend 分開呈現。
- `pytest` / `ruff` / schema validation：已完成。
- raw JSONL 重建 summary：validator 直接從 raw rows 重算。
- unsupported capability 不 crash matrix：adapter probe + explicit `RuntimeUnavailable`。
- correctness/quality fail 不進 ranking：`score_candidates` hard filter。
- policy 至少兩條 workload rule：有足夠 eligible rows 時 deterministic build；無 evidence 時拒絕。
- custom kernel fallback：未通過 GPU/dtype/shape cache 一律 reference。
- UI 所有正式數字追溯 raw run：production dashboard 只讀 SQLite artifacts；prototype 仍反覆標示 DEMO。

## Remaining formal experiment work

E00–E30 是實驗 catalog，不是可由 unit test 取代的 checkbox。正式完成仍需要：下載/授權模型、安裝 llama.cpp 與 vLLM 隔離環境、E01 timer calibration、E02 thermal block、E04–E09 cross-runtime runs、quality datasets、matched interventions、holdout replay，以及完整 Triton shape sweep和模型整合。這些工作產生的數字只有通過 G0–G8 才能進 README。
