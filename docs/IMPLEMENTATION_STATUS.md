# Implementation Status

更新日期：2026-08-04。此表把「工程能力完成」與「需要長時間正式實驗才能成立的研究結論」分開，避免以程式存在冒充實驗完成。

| Phase | Engineering | Validation on this checkout | Public claim status |
|---|---|---|---|
| M0 Trust the Clock | 完成 fingerprint、doctor、raw timing、warmup split、artifact hash、E00–E03 runner | RTX 4080 SUPER 正式 E00–E03 4/4 pass；E02 已用每 condition 1,000 次、30 秒 stabilization 與 background CUDA negative control 正式通過 7/7 gates | 無性能 headline |
| M1 Compare Fairly | 四 adapter contract；真正 tensor batch／HTTP concurrency；vLLM exact token-ID transport；固定版本 llama.cpp/vLLM 隔離環境與 loopback service control；runtime-specific pinned quality evaluator；E09 exact-scope fairness auditor | Llama scope 與 primary Ministral 3B 的 `p1024-o128` 各自完成 30 次 eligible cross-runtime audit；Ministral 報告明列 Q5_K_M eager 對 BF16 graph caveat | 僅為 exact-workload、quality-gated plan ordering；不宣稱純 runtime 因果效果 |
| M2 Explain Bottleneck | profiler schema、Nsight command、deterministic diagnosis | rule tests pass | diagnosis 僅 HYPOTHESIS |
| M3 Select by Workload | candidate pruning、session objective、policy/fallback/drift、E20 fresh-process cold/warm runner；E10 最大完整 plan 子矩形分析 | E06 正式 mixed-shape pass；E20 cached-host 30-pair scope pass；E10 graph-mbt32768 已有 35/45 policy-eligible buckets，仍 incomplete；E19 exact-scope break-even `PASS` | 不允許把單一 plan 的 35/45 外推成完整雙 plan policy 結論 |
| M4 Optimize Hot Path | fused residual+RMSNorm reference/Triton/correctness cache/measured-speedup dispatch/fallback、Llama integration/rollback | E22 216/216；E23 216-row randomized sweep全部 correctness pass；E24 30+30 paired search/holdout pass | E24 僅支持 Llama 3.2 3B BF16 的已量測 scope |
| M5 Publish Evidence | CLI、localhost API、typed worker queue、managed runtime、Local-first Web App、SQLite、research roadmap、verification、CI | Host/Origin/token/body/path/shell contracts；STARTING/RUNNING/STOPPED GPU 實機流程；桌面與窄版 UI 實機渲染；production UI 無 mock 數字 | release 可展示工程，不宣稱未量測加速 |
| M6 Learned component | E25/E27 unique-point 與 intervention-label exporter、prerequisite audit 已完成 | 目前 47/2,000 unique validated points、0/50 per-class intervention labels | 門檻前拒絕訓練，不以 repetitions 或重跑灌水 |
| M7 Copilot | deterministic grounded answer/citation/refusal pipeline，核心不依賴 Agent | E28 固定 22 題：22/22 checks pass、unsupported numeric claims 0 | 僅回答 artifact 可支持的 scope |

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

- `E02/result.json`：正式 thermal/background block，cold／stabilized／background CUDA 各 1,000 次、30 秒 stabilization。stabilized median 83.968 µs、robust CV 3.616%、latency drift 1.220%、matched temperature delta 0°C、active clock drift 0%；background median 187.904 µs（2.238×）且 negative control 被偵測，7/7 gates `PASS`。
- `run-20260804-021937-c18341`：Ministral 3 3B Q4_K_M／llama.cpp E07，30 measured rows，G1/G3/G5 pass，`CONDITIONAL_PASS`；quality artifact 與穩定 clock block 尚未完成。
- `run-20260804-022241-6f46a2`：SmolLM2 360M／vLLM E08 capability smoke，30 measured rows，G1/G3/G5 pass，`CONDITIONAL_PASS`；不是 primary 3B confirmatory result。
- `run-20260804-022845-aa4e8b`：Web App managed llama.cpp 的 start → authenticated inference job → stop 整合驗證，`CONDITIONAL_PASS`，停止後沒有殘留 GPU process。
- `run-20260804-023246-74791c`：Web App managed vLLM 的 start → authenticated inference job → stop 整合驗證，`CONDITIONAL_PASS`，WSL pinned-memory/native-sampler 路徑生效。
- `run-20260804-034044-369128`：Llama 3.2 3B／torch.compile default static／128→32／batch 1，30 次正式量測，正確性、BF16 reference quality、stability 與 G0–G8 全部通過，`PASS` 且可進 policy；此結論只適用該精確 scope。
- `quality/llama-3.2-3b-instruct-vllm-bf16.json`：vLLM exact-token 正式品質 gate。ARC-C 50 題 candidate/reference 同為 44%；WikiText perplexity 17.9717／17.9585（ratio 1.00074 ≤ 1.05），protocol match 且 `PASS`。
- `run-20260804-093759-0f0621`：Llama 3.2 3B BF16／vLLM 0.26、V1 model runner、eager、single-sequence、4096 scheduler bucket，`p1024-o128` 30 次正式量測。median latency 1946.00 ms、TTFT 70.87 ms、TPOT 14.76 ms、drift 1.31%、robust CV 7.92%，G0–G8 `PASS` 且可進 policy。V2 runner 的長 stall 與未達穩定門檻 probe 均未納入 eligible evidence。
- `run-20260804-094727-801034`：同 commit、同 workload 的 torch.compile `max-autotune-no-cudagraphs` dynamic confirmatory run，30 次；median latency 1711.20 ms、TTFT 90.64 ms、TPOT 12.75 ms、drift 2.09%、robust CV 7.07%，G0–G8 `PASS` 且可進 policy。
- `E09/fairness_audit.json`：上述兩個 run 的 model revision、workload、sampling、quality profile、30 repetitions 與 hardware fingerprint 全部 exact-match，狀態 `PASS`。此 scope 中 torch.compile median request latency 較低，vLLM median TTFT 較低；這只是描述性 ordering，不是跨 bucket 或跨模型推薦。
- `E06/result.json`：三個 dynamic mode 各 210 筆正式 observation。`dynamic=True` 為有效模式中最低 steady mixed-sequence median（1465.18 ms、robust CV 4.06%、2 unique graphs）；`dynamic=False` 雖穩定但 exact greedy output agreement 只有 75%，已排除且不會進 shape rule。
- `E24/result.json`：Llama 3.2 3B BF16 的 kernel-off/on 30 對 ABBA。search median 494.20 → 483.46 ms（2.22%，paired median-difference CI 12.87–172.79 ms），untouched holdout 824.53 → 700.06 ms（17.78%，CI 9.05–114.70 ms）；正確性通過，62,496 次 Triton dispatch、2,016 次校正式 fallback，狀態 `END_TO_END_SUPPORTED`。此結果不外推到其他模型、runtime 或 shape。
- `E20/result.json`：30 個 fresh Python process、Llama 3.2 3B BF16、128→8。cached-host time-to-first-usable median 4765.81 ms，模型/tokenizer load median 1973.82 ms；同程序 warmed response host median 115.09 ms，paired difference CI 4482.75–4655.55 ms，輸出與硬體 fingerprint 全數一致。此 scope 明確不涵蓋 machine reboot、dropped OS cache、persisted compile cache 或外部 runtime service restart。
- `E09/ministral-fairness-audit.json`：Ministral 3B 的 llama.cpp Q5_K_M 與 vLLM BF16 graph 在相同 `p1024-o128-c1`、30 repetitions、同一穩定硬體身分下 `PASS`；model format、revision、quantization 與 execution mode 差異均列為 caveat，`causal_runtime_isolation=false`。
- `E21/result.json`：10 次累積 CUDA profile iteration 的 profile-first selection `PASS`，下一週期候選為 `fused-rope-v1`；不反向替既有 E24 RMSNorm 結論背書。
- `E28/result.json`：固定 22 題 grounded question set 全部通過 exact citation、safe refusal 與 unsupported-numeric-claim gate。

這些 run 證明本機資料鏈可運作；E09 只允許上述單一 exact scope 的描述性排序，不構成一般化跨 runtime 勝負或部署推薦。

## Remaining formal experiment work

E00–E30 是實驗 catalog，不是可由 unit test 取代的 checkbox。primary Ministral 3B 的 llama.cpp Q5_K_M 與 vLLM BF16 quality、正式 `p1024-o128-c1` runs 與 E09 exact-scope audit已成立；Q4_K_M 因 ARC-C drop 超過 1pp 保留為 negative evidence。E10 graph-mbt32768 profile 現有 35/45 policy-eligible buckets；p2048-o512-c8 與 p4096 的 9 格尚待續跑，第二個 graph-mbt65536 profile 尚未開始，因此整體仍是 `INCOMPLETE`。E11–E18 仍需要 real-distribution replay、controlled mediator、negative-control 與 holdout pairs。E21 profile-first candidate evidence 已 `PASS`。E25 目前只有 47/2000 unique validated points，E26 因 prerequisite 未達而拒絕訓練；E27 尚無每類 50 筆正式 intervention labels。E28 固定 22 題 grounding test 已 `PASS`。E29 還需要至少 5 位同領域使用者完成 usability tasks；E30 需要另一台機器或真正乾淨環境重現。只有通過 G0–G8 的數字才能進 README。
