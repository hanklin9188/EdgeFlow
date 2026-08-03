# Implementation Status

更新日期：2026-08-04。此表把「工程能力完成」與「需要長時間正式實驗才能成立的研究結論」分開，避免以程式存在冒充實驗完成。

| Phase | Engineering | Validation on this checkout | Public claim status |
|---|---|---|---|
| M0 Trust the Clock | 完成 fingerprint、doctor、raw timing、warmup split、artifact hash、E00–E03 runner | RTX 4080 SUPER 正式 E00–E03 4/4 pass；timer negative control 與 thermal/background policy 已保存 | 無性能 headline |
| M1 Compare Fairly | 四 adapter contract；真正 tensor batch／HTTP concurrency；HTTP 混合 prompt 分布；固定版本 llama.cpp/vLLM 隔離環境與 loopback service control；pinned quality reference evaluator；E09 exact-scope fairness auditor | llama.cpp primary Q4_K_M、vLLM SmolLM2 各完成 30-repetition smoke；PyTorch batch=2 GPU smoke pass；不相容 run 會拒絕產生排序 | primary model 跨 runtime 結論未建立 |
| M2 Explain Bottleneck | profiler schema、Nsight command、deterministic diagnosis | rule tests pass | diagnosis 僅 HYPOTHESIS |
| M3 Select by Workload | candidate pruning、session objective、policy/fallback/drift、E20 fresh-process cold/warm runner | E06 正式 mixed-shape pass；E20 cached-host 30-pair scope pass | E10/E19 仍需共同 exact-scope eligible rows |
| M4 Optimize Hot Path | fused residual+RMSNorm reference/Triton/correctness cache/measured-speedup dispatch/fallback、Llama integration/rollback | E22 216/216；E23 216-row randomized sweep全部 correctness pass；E24 30+30 paired search/holdout pass | E24 僅支持 Llama 3.2 3B BF16 的已量測 scope |
| M5 Publish Evidence | CLI、localhost API、typed worker queue、managed runtime、Local-first Web App、SQLite、research roadmap、verification、CI | Host/Origin/token/body/path/shell contracts；STARTING/RUNNING/STOPPED GPU 實機流程；桌面與窄版 UI 實機渲染；production UI 無 mock 數字 | release 可展示工程，不宣稱未量測加速 |
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
- `run-20260804-034044-369128`：Llama 3.2 3B／torch.compile default static／128→32／batch 1，30 次正式量測，正確性、BF16 reference quality、stability 與 G0–G8 全部通過，`PASS` 且可進 policy；此結論只適用該精確 scope。
- `E06/result.json`：三個 dynamic mode 各 210 筆正式 observation。`dynamic=True` 為有效模式中最低 steady mixed-sequence median（1465.18 ms、robust CV 4.06%、2 unique graphs）；`dynamic=False` 雖穩定但 exact greedy output agreement 只有 75%，已排除且不會進 shape rule。
- `E24/result.json`：Llama 3.2 3B BF16 的 kernel-off/on 30 對 ABBA。search median 494.20 → 483.46 ms（2.22%，paired median-difference CI 12.87–172.79 ms），untouched holdout 824.53 → 700.06 ms（17.78%，CI 9.05–114.70 ms）；正確性通過，62,496 次 Triton dispatch、2,016 次校正式 fallback，狀態 `END_TO_END_SUPPORTED`。此結果不外推到其他模型、runtime 或 shape。
- `E20/result.json`：30 個 fresh Python process、Llama 3.2 3B BF16、128→8。cached-host time-to-first-usable median 4765.81 ms，模型/tokenizer load median 1973.82 ms；同程序 warmed response host median 115.09 ms，paired difference CI 4482.75–4655.55 ms，輸出與硬體 fingerprint 全數一致。此 scope 明確不涵蓋 machine reboot、dropped OS cache、persisted compile cache 或外部 runtime service restart。

這些 run 證明本機資料鏈可運作，不構成跨 runtime 勝負或部署推薦。

## Remaining formal experiment work

E00–E30 是實驗 catalog，不是可由 unit test 取代的 checkbox。E00–E06、E20、E22–E24 已有正式本機 artifact。Llama 3.2 3B 的 E04 12-case eager matrix 已完整執行：1 個 `PASS`／policy-eligible、11 個因 Windows/WSL 共用 GPU 的 G4 stability 為 `CONDITIONAL_PASS`，0 failure／0 prune；固定 bucket 已重用 exact token IDs，telemetry 也移到同步 engine timer 關閉後。E05 32-case 矩陣已完整執行：16 個支援 scope 中 6 個 `PASS`／policy-eligible、10 個 `CONDITIONAL_PASS`，另 16 個會啟用內部 CUDAGraph 的組合已明確 capability-pruned。矩陣案例各自在隔離子程序執行；即使 CUDA／PyTorch 發生 SIGSEGV，父程序仍會封存 partial artifact、標記失敗並繼續，失敗 execution 也不可能因殘留 metrics 誤通過 validation。E06 已選出 correctness/stability 通過的 dynamic rule；E20 cached-host fresh-process scope 與 E24 search/untouched-holdout 模型端驗證均已正式通過。固定版本 llama.cpp 與 vLLM 已安裝並通過 capability smoke，但 primary Ministral 3 3B 的 E07–E09 same-scope BF16/quant quality 與 cross-runtime 正式矩陣仍未成立。E10/E19 strict analysis 正確拒絕不完整共同 scope；E11–E18 仍需要 real-distribution replay、controlled mediator、negative-control 與 holdout pairs。E20 的真正 dropped OS cache、persisted compile cache 與外部 service restart 仍分開列為未驗證 scope。E21 必須補 profile-first candidate evidence，不能用已完成 kernel 反向杜撰選擇理由。readiness audit 顯示 E25 目前只有 7/2000 unique validated points、E27 尚無正式 intervention labels、E28 尚無固定 grounded question set，因此 E25–E28 維持 prerequisite-blocked。E29 還需要至少 5 位同領域使用者完成 usability tasks；E30 需要另一台機器或真正乾淨環境重現。這些工作產生的數字只有通過 G0–G8 才能進 README。
