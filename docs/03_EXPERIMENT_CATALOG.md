# 03 · Experiment Catalog

本文件是實際執行清單。每個 experiment 都包含 hypothesis、變因、程序、驗收、產物與失敗解讀。編號不可重用；取消的實驗保留狀態。

---

# Phase 0 · Environment and Measurement Integrity

## E00 — Hardware and Software Fingerprint

**目的**：確保每個結果可以綁定到確切環境。

**輸入**：目前工作站。

**程序**：

1. 收集 GPU、driver、CUDA、CPU、RAM、OS／WSL。
2. 收集 PyTorch、Triton、Transformers、vLLM、llama.cpp、Nsight版本。
3. 執行 `nvidia-smi`、PyTorch CUDA、Triton vector-add、llama.cpp CUDA probe、vLLM import。
4. 計算 config 與 executable hash。

**驗收**：

- GPU 顯示 RTX 4080 SUPER；
- PyTorch 能配置 CUDA tensor；
- Triton kernel correctness pass；
- 每個 backend capability 有明確 `verified/failed/skipped`；
- fingerprint JSON 符合 schema。

**產物**：`hardware_fingerprint.json`、`capabilities.json`。

**失敗解讀**：工具不可用是環境問題，不是 performance 結論。

---

## E01 — Timer Calibration

**Hypothesis**：CUDA Event、`perf_counter_ns` 與 end-to-end wall time 在正確同步後會有一致排序，但絕對邊界不同。

**變因**：

- timer：CUDA Event／wall clock；
- sync：before+after／after only／none；
- workload：sleep-like CPU、CUDA vector op、small GEMM。

**程序**：

- 每組 1,000 iterations；
- 比較 timer bias、variance；
- 故意執行錯誤同步作 negative control。

**驗收**：

- EdgeFlow 正式 engine timer 使用 documented boundary；
- 無同步方式被測出低估且標記禁止；
- overhead 可量化。

---

## E02 — Thermal and Background Noise Study

**Hypothesis**：未穩定溫度或有背景 GPU process 時，latency variance 顯著提高。

**條件**：

1. cold GPU；
2. stabilized GPU；
3. browser video／背景 CUDA negative control；
4. 長時間連續負載。

**指標**：clock、temperature、power、median latency、CV、前後半 drift。

**驗收**：建立正式 precondition threshold，例如：

- background utilization < 5%；
- clock drift < 3%；
- matched pair temperature差 < 5°C。

**產物**：`measurement_policy.yaml`。

---

## E03 — Repetition Count and Confidence Stability

**目的**：決定 10／20／30／50 repetition 的 CI 穩定性。

**程序**：從一個 200-repetition reference run 做 repeated subsampling。

**指標**：median error、CI width、winner reversal rate。

**驗收**：選擇能讓 winner reversal < 5% 的最小 repetitions；不同實驗類型可使用不同值。

---

# Phase 1 · Backend Baselines

## E04 — PyTorch Eager Baseline

**模型**：主要 3B 模型；BF16。

**矩陣**：prompt 128／1024／4096，output 32／128，batch 1／4。

**指標**：load、TTFT、TPOT、tok/s、VRAM。

**控制**：固定 tokenizer、prompt token IDs、greedy、KV cache。

**驗收**：

- 30 repeated request；
- correctness reference artifacts；
- no profiler 正式 run；
- 建立後續比較基線。

---

## E05 — `torch.compile` Mode Sweep

**Hypothesis**：最佳 compile mode 依 workload 與 session horizon 改變。

**變因**：

- eager；
- `default`；
- `reduce-overhead`；
- `max-autotune`；
- `max-autotune-no-cudagraphs`；
- dynamic `False/None/True`；
- `fullgraph` diagnostic。

**量測**：

- first compile time；
- subsequent compile/recompile count；
- steady TTFT/TPOT；
- peak VRAM；
- graph breaks；
- session cost N=1/5/20/100/1000。

**驗收**：

- compile cost與steady-state分開；
- graph break report；
- 至少找出一個「steady winner != short-session winner」案例，若不存在也要誠實報告。

**失敗解讀**：compile error 不代表 PyTorch 不可用；plan 標 unsupported。

---

## E06 — Dynamic Shape and Recompilation

**Hypothesis**：混合 prompt length 會造成 shape specialization／recompilation，影響 compiled plan。

**序列**：

```text
128 → 128 → 1024 → 128 → 2048 → 1024 → 4096
```

**比較**：dynamic false／auto／true、shape bucket policy。

**指標**：compile count、compile time、latency spike、cache reuse、VRAM。

**驗收**：產生 shape-bucket rule；不能只在固定 shape 宣稱 compile benefit。

---

## E07 — llama.cpp Quantization Sweep

**模型**：同一 official base/instruct checkpoint 的 GGUF：F16/BF16（可行時）、Q8_0、Q6_K、Q5_K_M、Q4_K_M。

**變因**：

- quantization；
- prompt 128／1024／4096；
- generation 32／128；
- batch／ubatch；
- flash attention on/off；
- GPU layers all；
- KV type（支援時）。

**指標**：pp/tg、TTFT、TPOT、VRAM、load、quality。

**品質**：WikiText-2 + ARC-C subset，通過後再進完整 quality。

**驗收**：建立 latency-memory-quality Pareto，而不是預設 Q4 最好。

---

## E08 — vLLM Scheduling Sweep

**Hypothesis**：`max_num_batched_tokens` 與 concurrency 改變 TTFT/ITL/throughput trade-off。

**變因**：

- eager／CUDA Graph支援設定；
- max_num_batched_tokens：1024／2048／4096／8192（受VRAM限制）；
- max_num_seqs：1／4／8／16；
- concurrency：1／2／4／8；
- chunked prefill current default／explicit settings。

**工作負載**：短 prompt mix、長 prompt mix、mixed prefill-decode。

**指標**：TTFT、ITL、throughput、queue、preemption／OOM。

**驗收**：找出至少兩個 workload bucket 的不同最佳 token budget，或證明在本硬體範圍內相同。

---

## E09 — Cross-Runtime Fairness Audit

**目的**：確認 PyTorch、llama.cpp、vLLM 比的是相同任務。

**核對**：

- tokenizer revision；
- prompt token IDs；
- chat template；
- BOS/EOS；
- max output；
- greedy semantics；
- KV cache；
- weight source與quantization；
- engine-only timing boundary。

**驗收**：產生 `fairness_audit.md`；不公平 pair 不進 headline comparison。

---

# Phase 2 · Workload-Conditioned Policy

## E10 — Fixed Plan Dominance Test

**RQ**：是否存在一個固定 plan 支配所有 workload？

**候選**：每 backend 1–3 個 validated Pareto plan。

**workload grid**：

- prompt 128／512／1024／2048／4096；
- output 32／128／512；
- concurrency 1／4／8。

**指標**：normalized objective、per-bucket winner、regret。

**驗收**：

- 計算每個 fixed plan 的 expected cost；
- 計算 oracle per-bucket policy；
- 若 oracle gain > practical threshold，支持 conditioned policy motivation。

---

## E11 — Policy Synthesis Baselines

**方法**：

1. single global winner；
2. hand rule；
3. decision tree on measured features；
4. nearest-neighbor plan；
5. EdgeFlow evidence-constrained policy。

**split**：workload bucket holdout，不可 random row leakage。

**指標**：

- expected objective；
- p95 latency；
- quality violation；
- memory violation；
- policy complexity；
- fallback rate；
- regret to oracle。

**驗收**：EdgeFlow policy 至少不劣於最強簡單 baseline，且規則可解釋。

---

## E12 — Real-Distribution Replay

**資料**：UltraChat 200k；LMSYS-Chat-1M 可選 gated。

**處理**：只使用 token length／turn distribution與合法 prompt sample，固定 seed。

**比較**：fixed winner vs conditioned policy。

**指標**：session total latency、TTFT p95、TPOT、plan switches、fallback。

**驗收**：controlled grid 的 gain 需在 real-distribution replay 保持方向；若下降，分析 distribution shift。

---

# Phase 3 · Causal Diagnosis

## E13 — Launch-Overhead Intervention

**Observation trigger**：

- median kernel duration < 15 μs；
- kernel gap ratio > 15%；
- GPU utilization偏低；
- CPU launch thread繁忙。

**Interventions**：

- `reduce-overhead`；
- CUDA Graph；
- operator fusion/custom kernel；
- batch increase negative/positive control。

**預期中介結果**：kernel gap下降、launch count下降。

**接受 hypothesis**：

- 中介指標按預期改善；
- unprofiled TPOT lower CI 至少改善 2%；
- correctness pass。

若 latency改善但 gap不變，不能宣稱 launch causal path。

---

## E14 — Memory-Bandwidth Intervention

**Trigger**：decode階段高 DRAM traffic、低 compute utilization、quantization顯著影響。

**Interventions**：

- BF16 → Q8/Q6/Q4；
- batch increase；
- weight format／kernel變更；
- KV-cache dtype（支援時）。

**中介指標**：bytes/token、DRAM throughput、L2 hit、kernel duration。

**控制**：同模型語意與 quality gate。

**驗收**：若量化變快但 quality超標，只可歸類 memory-first，不可推薦 strict profile。

---

## E15 — Compute-Bound Prefill Intervention

**Trigger**：長 prompt、Tensor Core active、GEMM dominates、low gap。

**Interventions**：

- compile max-autotune；
- shape padding；
- batch調整；
- attention backend／flash attention；
- precision。

**指標**：prompt tok/s、SM utilization、tensor throughput、TTFT。

**驗收**：diagnosis 必須區分 prefill 與 decode，不得以整體平均掩蓋。

---

## E16 — KV-Cache Capacity and Long Context

**目標**：建立 16GB VRAM 的 context／concurrency capacity frontier。

**矩陣**：prompt 1024／2048／4096／8192；concurrency 1／2／4／8；不同 backend/KV dtype。

**指標**：peak VRAM、max successful requests、OOM、TTFT、TPOT。

**驗收**：輸出 capacity map 與 conservative safety margin（例如保留 5–10% VRAM）。

---

## E17 — Scheduler-Bound Mixed Workload

**工作負載**：同時存在長 prefill與短 decode request。

**比較**：vLLM token budgets／llama.cpp continuous batching／簡單 EdgeFlow queue。

**指標**：queue delay、TTFT p95、ITL p95、throughput、starvation。

**驗收**：若 throughput增益以嚴重尾延遲換取，UI 必須顯示 trade-off，不能只標 winner。

---

## E18 — Negative-Control Diagnosis

**目的**：測試 diagnoser 是否會過度解讀。

**方法**：對已知 compute-heavy case 套用 launch intervention；對 launch-heavy case只改不相關參數。

**指標**：false-positive rate、unsupported hypothesis rate。

**驗收**：diagnoser 必須能輸出 `insufficient_evidence`，而不是每次都給肯定答案。

---

# Phase 4 · Amortization

## E19 — Session Break-Even Study

**候選**：eager、compile modes、llama.cpp quant、vLLM service。

**N**：1／5／20／100／1000。

**成本**：process start、model load、compile、capture、request。

**輸出**：

\[
N_{A\rightarrow B}=\frac{C_{startup,A}-C_{startup,B}}{C_{req,B}-C_{req,A}}.
\]

**驗收**：生成每一 pair 的 break-even；無交點時明示。

---

## E20 — Cold Start vs Warm Restart

**條件**：

- OS file cache cold（可近似，不宣稱完全冷）；
- process cold；
- model cached；
- compile cache persisted；
- long-lived service。

**指標**：time-to-first-usable-response。

**驗收**：README 不以 warm start 數字暗示首次使用體驗。

---

# Phase 5 · Custom Kernel

## E21 — Kernel Candidate Selection

**原則**：先 profile，再選 kernel。候選：

1. fused RMSNorm + residual；
2. fused SiLU-mul epilogue；
3. fused residual + gated activation；
4. rank-small adapter epilogue（未整合 FAD 前可用 synthetic module）；
5. sampling/top-k小 kernel。

**評分**：

\[
S = \text{time share}\times\text{launch frequency}\times\text{fusion opportunity}\times\text{implementation feasibility}.
\]

**驗收**：選擇結果必須附 profile evidence；不能先寫 kernel 再找理由。

---

## E22 — Triton Kernel Correctness Sweep

**shape**：

- hidden 1024／2048／3072／4096；
- tokens 1／4／16／128／512；
- dtype FP32／FP16／BF16；
- contiguous／selected stride；
- edge shapes非2次方。

**測試**：forward、NaN/Inf、random seeds、extreme values。

**驗收**：dtype-specific `assert_close`；unsupported shape明確 fallback。

---

## E23 — Triton Kernel Performance Sweep

**baselines**：PyTorch eager、compiled equivalent、Triton。

**程序**：warmup、100+ iterations、CUDA events、randomized shape order。

**輸出**：speedup heatmap、GB/s／FLOP estimate、faster/slower region。

**驗收**：不得只報最佳 shape；需定義 dispatch rule與最低實用 speedup。

---

## E24 — End-to-End Kernel Integration

**目的**：證明 microbenchmark 改善能轉成模型 inference benefit。

**比較**：kernel off/on；完全相同模型／prompt／plan。

**指標**：kernel launch count、GPU time share、TTFT/TPOT、VRAM、correctness。

**接受**：end-to-end lower CI ≥ 2% 或明確改善尾延遲／記憶體；否則誠實標示 micro-only contribution。

---

# Phase 6 · Optional Learned Components

## E25 — Cost Model Dataset Construction

**來源**：所有 validation-pass runs 的 aggregated configuration records。

**最小量**：建議 ≥ 2,000 unique plan-workload points；重複 run 不算 unique。

**features**：hardware、model architecture、plan、prompt/output、concurrency、trace summary。

**targets**：TTFT、TPOT、VRAM、throughput、failure probability。

**驗收**：資料 split 防止同 workload 重複洩漏。

---

## E26 — Cost Model Evaluation

**baseline**：linear、nearest neighbor、random forest／gradient boosting、analytical memory model。

**指標**：MAPE、Spearman、top-k recall、selection regret、constraint violation。

**使用邊界**：只用於 candidate pruning；final winner 必須實測。

---

## E27 — Bottleneck Classifier

**label**：由 intervention結果建立，而不是只靠人工看 trace。

類別：launch、memory、compute、KV capacity、scheduler、compile、I/O、insufficient evidence。

**驗收**：group split，macro-F1、calibration；若資料不足，保留 rule engine。

---

# Phase 7 · Optional Copilot and UI

## E28 — Copilot Grounding Test

**輸入**：固定 run records與問題。

**測試**：是否引用正確數字、區分 measured/inferred/hypothesis、拒絕缺資料。

**指標**：numeric fidelity、citation coverage、unsupported claim rate。

**門檻**：unsupported numeric claim = 0；否則不公開啟用。

---

## E29 — UI Usability Audit

**任務**：

1. 找出某 workload 最佳 validated plan；
2. 找出一項 invalid result 的原因；
3. 從 summary 回到 raw run；
4. 比較 cold與steady winner；
5. 判斷一項 hypothesis 是否被支持。

**指標**：task success、time、錯誤理解、主觀清晰度。

**最低樣本**：5 位同領域使用者即可作 portfolio usability test；完整研究需更多。

---

## E30 — Reproducibility Challenge

邀請另一台機器或乾淨環境執行：

- setup；
- smoke model；
- 3個 benchmark；
- report rebuild。

**驗收**：步驟無隱藏路徑、結果格式一致；數值可不同但趨勢與環境差異可解釋。

---

# Headline Claim 對應實驗

| Claim | 必要實驗 |
|---|---|
| Workload-conditioned policy 優於固定 plan | E10–E12 |
| Causal search 更省 trial | E13–E18 + search baseline |
| Amortization 改變 deployment winner | E19–E20 |
| Custom kernel 有系統價值 | E21–E24 |
| Optional learned model 有效 | E25–E27 |
| Copilot 不亂編數字 | E28 |
| GitHub 展示可理解 | E29–E30 |
