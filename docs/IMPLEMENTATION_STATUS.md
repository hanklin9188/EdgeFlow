# Implementation Status

更新日期：2026-08-04。此表把「工程能力完成」與「需要長時間正式實驗才能成立的研究結論」分開，避免以程式存在冒充實驗完成。

| Phase | Engineering | Validation on this checkout | Public claim status |
|---|---|---|---|
| M0 Trust the Clock | 完成 fingerprint、doctor、raw timing、warmup split、artifact hash | unit/integration pass；RTX 4080 SUPER 可用 | 無性能 headline |
| M1 Compare Fairly | 四 adapter contract；固定版本 llama.cpp/vLLM 隔離環境與 loopback service control | llama.cpp primary Q4_K_M、vLLM SmolLM2 各完成 30-repetition smoke evidence | primary model 跨 runtime 結論未建立 |
| M2 Explain Bottleneck | profiler schema、Nsight command、deterministic diagnosis | rule tests pass | diagnosis 僅 HYPOTHESIS |
| M3 Select by Workload | candidate pruning、session objective、policy/fallback/drift | policy tests pass | 需 formal eligible rows + holdout |
| M4 Optimize Hot Path | fused residual+RMSNorm reference/Triton/cache/fallback | GPU correctness command可執行 | end-to-end claim 未建立 |
| M5 Publish Evidence | CLI、localhost API、typed worker queue、managed runtime、Local-first Web App、SQLite、verification、CI | Host/Origin/token/body/path/shell contracts；STARTING/RUNNING/STOPPED GPU 實機流程；production UI 無 mock 數字 | release 可展示工程，不宣稱未量測加速 |
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
- Local-first control：CLI拒絕非 loopback bind；寫入需 session token；一次一個 GPU subprocess/runtime；runtime 使用 loopback 與隨機 API key；command/path/environment 不由 browser 提供。

## Current RTX evidence

- `run-20260804-021937-c18341`：Ministral 3 3B Q4_K_M／llama.cpp E07，30 measured rows，G1/G3/G5 pass，`CONDITIONAL_PASS`；quality artifact 與穩定 clock block 尚未完成。
- `run-20260804-022241-6f46a2`：SmolLM2 360M／vLLM E08 capability smoke，30 measured rows，G1/G3/G5 pass，`CONDITIONAL_PASS`；不是 primary 3B confirmatory result。
- `run-20260804-022845-aa4e8b`：Web App managed llama.cpp 的 start → authenticated inference job → stop 整合驗證，`CONDITIONAL_PASS`，停止後沒有殘留 GPU process。
- `run-20260804-023246-74791c`：Web App managed vLLM 的 start → authenticated inference job → stop 整合驗證，`CONDITIONAL_PASS`，WSL pinned-memory/native-sampler 路徑生效。

這些 run 證明本機資料鏈可運作，不構成跨 runtime 勝負或部署推薦。

## Remaining formal experiment work

E00–E30 是實驗 catalog，不是可由 unit test 取代的 checkbox。固定版本 llama.cpp 與 vLLM 已安裝並通過 capability smoke；正式完成仍需要：E01 timer calibration、E02 thermal block、primary 3B 的 E04–E09 cross-runtime runs、quality datasets、matched interventions、holdout replay，以及 Triton 模型整合。完整 RMSNorm shape sweep 已在本機通過，但尚未建立 end-to-end headline。E29 還需要至少 5 位同領域使用者完成 usability tasks；E30 需要另一台機器或真正乾淨環境重現。這些工作產生的數字只有通過 G0–G8 才能進 README。
