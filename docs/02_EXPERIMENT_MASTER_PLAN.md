# 02 · Experiment Master Plan

## 2.1 實驗總目標

EdgeFlow 的實驗不是為了找一張「最高 tokens/s」表，而是要驗證四件事：

1. **Measurement validity**：量測是否公平、穩定、可重現。
2. **Conditional optimality**：最佳 execution plan 是否隨 workload 改變。
3. **Causal validity**：diagnosis 提出的瓶頸是否能被 intervention 支持或否定。
4. **Deployment utility**：包含啟動與品質成本後，EdgeFlow policy 是否降低真實使用成本。

---

## 2.2 研究問題

### RQ1 — 固定 plan 是否足夠？

> 在同一模型與 GPU 上，是否存在一個 plan 能同時支配短 prompt、長 prompt、低 concurrency 與高 concurrency？

若不存在，workload-conditioned policy 才有必要。

### RQ2 — profiler-guided search 是否更有效率？

> 相較 grid/random/TPE，EdgeFlow 是否用更少 trial 找到相近或更好的 plan？

### RQ3 — 啟動成本是否改變最佳選擇？

> 只看 steady-state TPOT 的 winner，是否在短 session 反而更慢？

### RQ4 — 診斷是否可驗證？

> 對 launch、memory、compute、scheduler、KV-cache 等候選 bottleneck，對應 intervention 是否改變預期的中介指標與最終 latency？

### RQ5 — custom kernel 是否有端到端價值？

> microbenchmark speedup 是否能跨 shape、跨模型並轉換成 end-to-end 改善？

### RQ6 — quality constraint 是否改變 Pareto frontier？

> Q4/Q5/Q6/Q8 與 BF16 的 latency-memory-quality trade-off 如何影響 recommendation？

---

## 2.3 固定硬體範圍

第一篇完整作品以一張 GPU 為主：

- NVIDIA GeForce RTX 4080 SUPER。
- 16 GB GDDR6X。
- 單 GPU。
- 不超頻；若使用 factory OC，記錄實際 board model 與 power limit。
- Windows 11 + WSL2 Ubuntu，或 native Ubuntu。

正式結果必須附：

- board vendor／VBIOS；
- driver；
- CUDA；
- power limit；
- CPU／RAM；
- WSL/native；
- ambient／GPU temperature window；
- background process policy。

不能把其他硬體數字混入主結論；後續擴充才做 cross-hardware transfer。

---

## 2.4 環境基線

### 建議軟體 lane

| Lane | 用途 | 原則 |
|---|---|---|
| `stable` | 正式結果 | pin 所有版本與 commit |
| `latest` | 相容性／回歸 | 定期測最新穩定版，不取代 stable |
| `dev` | kernel／patch | 允許 editable install，不進主表 |

首次 implementation 時鎖定：

```text
Python 3.12
PyTorch stable CUDA wheel
Transformers compatible with selected models
Triton bundled/pinned with PyTorch
llama.cpp exact git commit
vLLM exact release and image digest
Nsight Systems exact version
Nsight Compute exact version
```

由於 2026 年工具仍快速更新，本規格不把套件 patch version 寫死；正式 repo 建立時應生成：

- `requirements.lock`；
- `environment.yml` 或 `uv.lock`；
- Docker image digest；
- `third_party_commits.yaml`。

---

## 2.5 Workload 空間

### A. Controlled token buckets

#### Prompt length

```text
32, 128, 512, 1024, 2048, 4096, 8192 tokens
```

8192 僅在模型與 VRAM 支援時執行；OOM 不等同失敗，而是 capacity boundary。

#### Output length

```text
1, 32, 128, 512 tokens
```

用途：

- `1`：prefill／first-token 主導；
- `32`：短 agent action；
- `128`：一般回答；
- `512`：長 generation。

#### Batch / concurrency

```text
batch_size: 1, 2, 4, 8, 16
concurrency: 1, 2, 4, 8
```

不是所有 backend 都有相同 batch 語意，因此分成：

- offline batch；
- closed-loop online concurrency；
- open-loop arrival rate。

### B. Arrival pattern

1. **Closed-loop**：每個 client 完成後才送下一個 request。
2. **Poisson**：到達間隔服從 exponential distribution。
3. **Burst**：固定時間一次送入 4／8 requests。
4. **Trace replay**：由真實或合成 session timestamp 重播。

### C. Session horizon

```text
N = 1, 5, 20, 100, 1000 requests
persistent service = 30 minutes or fixed 5000 requests
```

用來計算 load／compile／capture 成本能否攤平。

---

## 2.6 Prompt 建立規則

### Synthetic prompt

目的：精確控制 token 數與內容熵，避免 dataset 分布混入 shape experiment。

程序：

1. 從固定、無敏感內容的文字片段庫抽樣。
2. 使用目標模型 tokenizer 拼接。
3. 修剪或補齊到目標 token 數。
4. 重新 tokenize，要求誤差為 0 token。
5. prompt hash 與 token IDs 儲存。

不允許用重複單一 token 填滿全部 prompt，因為可能產生不代表一般文本的 attention／cache 行為。建議使用 8–32 段不同語句循環，並另設 `repeated-token stress` 作為特殊測試。

### Real prompt

從 UltraChat／LMSYS 等資料集中：

- 僅抽 user-visible prompt；
- 依 tokenizer token length 分層；
- 去除空白、極端異常、明顯個資；
- 不公開 gated dataset 原文；只公開 row ID、length、hash 與處理程式；
- 主結果以 synthetic controlled matrix 為準，real prompt 用於外部效度。

---

## 2.7 Generation 設定

### Performance protocol

```text
greedy decoding
sampling disabled
temperature = 0
fixed max_new_tokens
early EOS disabled when exact output-length control is required
```

如果 backend 無法完全關閉 EOS：

- 使用不容易提前結束的 prompt；或
- 以 forced token benchmark API；或
- 將實際 output length 作為 covariate，並另報 fixed-output subset。

### Quality protocol

依模型官方 chat template；允許 EOS。Sampling 必須固定，預設 greedy。Qwen3.5 主要 latency protocol關閉 thinking，另設 reasoning-mode experiment，不將兩者混合。

---

## 2.8 指標定義

### TTFT

\[
\mathrm{TTFT}=t_{\text{first token emitted}}-t_{\text{request accepted}}.
\]

需另報 engine-only 與 end-to-end。

### TPOT

\[
\mathrm{TPOT}=\frac{t_{\text{last}}-t_{\text{first}}}{N_{\text{out}}-1}.
\]

`N_out=1` 時不計 TPOT。

### ITL

\[
\mathrm{ITL}_i=t_i-t_{i-1}.
\]

報 median、p90、p95、p99 與最大 stall。

### Throughput

- prompt tok/s；
- generation tok/s；
- aggregate output tok/s；
- request/s。

不能以 aggregate throughput 取代 per-request latency。

### Memory

- static model bytes；
- CUDA allocated／reserved；
- peak VRAM；
- KV-cache estimate與實測；
- process RSS；
- OOM boundary。

### Startup

- process start；
- import；
- model load；
- graph compile；
- autotune；
- CUDA Graph capture；
- first usable request。

### Quality

- WikiText-2 perplexity；
- ARC-Challenge accuracy；
- GSM8K exact answer；
- HumanEval pass@1（extended）；
- greedy token agreement／logit divergence（同精度 correctness）。

---

## 2.9 正式量測程序

每個 configuration：

1. 系統 precheck。
2. 確認沒有其他 GPU process；若有，記錄並停止正式 run。
3. 等待 GPU utilization 低於門檻。
4. 取得 idle temperature／clock。
5. 啟動隔離 subprocess。
6. 載入模型，記錄 load time。
7. warmup：至少 5 requests，或直到最近 5 次 median drift < 2%。
8. compile/capture 另行記錄。
9. 正式 repetitions：
   - microbenchmark ≥ 100 measured iterations；
   - engine single request ≥ 30 repetitions；
   - online serving 每 profile ≥ 200 completed requests；
   - quality 依完整或固定 subset。
10. 每次 timing 前後使用正確 CUDA synchronization／event boundary。
11. 記錄 per-request raw events，不只 summary。
12. 收集 temperature、clock、power sampling。
13. run 結束後 validation。
14. 立即寫 raw artifact；summary 可重建。

---

## 2.10 Randomization 與 blocking

避免時間順序造成 thermal／cache bias：

- matched A/B 的 run order 以 seed randomize；
- 以 `ABBA` 或 randomized block 執行；
- backend 大型 load 間可安排 cooldown；
- 同一比較的 plans 必須在同一 driver／commit block；
- 若軟體版本不同，標記為 version study，不能當純 plan A/B。

建議 block：

```text
block = hardware + driver + CUDA + model revision + workload seed + day/session
```

---

## 2.11 Thermal 與 power 控制

RTX 消費級 GPU 容易受溫度與 power state 影響。

正式 protocol：

- 不手動超頻。
- power limit 固定並記錄。
- benchmark 前執行 2–5 分鐘穩定負載或直到 clock 穩定。
- run 期間每秒記錄 temperature、power、SM/memory clock。
- 若前半與後半 latency 差異 > 3%，標為 drift。
- GPU 溫度跨 matched pair 差異建議 ≤ 5°C。
- Windows 背景遊戲、影片轉碼、瀏覽器 WebGPU 不得與正式 run 同時進行。

---

## 2.12 統計規格

### Summary

- median 作為主要 latency estimate；
- mean、std、MAD、p95 作輔助；
- throughput 報 mean + bootstrap CI；
- raw distribution 可視化。

### Paired comparison

相同 prompt／seed／order 下配對。計算：

\[
\Delta_i = y_{A,i}-y_{B,i}.
\]

以 paired bootstrap 估 95% CI；至少 10,000 bootstrap resamples。

### Practical significance

僅統計顯著不夠。預設接受 performance improvement 條件：

- latency lower CI 對應至少 2% 改善；或
- throughput lower CI 至少 2%；
- microkernel 可用 3–5% 門檻，因為整合成本更高；
- end-to-end 若 < 2%，標為 neutral，不宣稱加速。

### Multiple comparison

大量 plan ranking 不逐一做顯著性宣稱。先：

1. search set 找 candidate；
2. holdout workload 進行 confirmatory comparison；
3. 只對預先指定 pair 做 CI。

---

## 2.13 Quality gate

### 同精度 backend correctness

- NaN/Inf = 0。
- greedy next-token agreement 高於預設門檻。
- logits／hidden comparison 使用 dtype-specific tolerance。
- 首次 mismatch 必須保存 prompt、token position、top-k logits。

### 量化 quality

預設 deployment profiles：

| Profile | Quality constraint |
|---|---|
| strict | ARC-C drop ≤ 0.5 pp，PPL increase ≤ 2% |
| balanced | ARC-C drop ≤ 1.0 pp，PPL increase ≤ 5% |
| memory-first | ARC-C drop ≤ 2.0 pp，PPL increase ≤ 10% |

這些是初始 policy threshold，不是宣稱所有模型都適用；UI 必須允許調整。

---

## 2.14 Search baseline

為驗證 EdgeFlow search：

- exhaustive grid（可行的小空間）；
- random search；
- TPE／Bayesian optimization；
- rule-only heuristic；
- EdgeFlow profiler-guided causal search；
- optional learned cost model pruning。

公平條件：

- 同一 candidate space；
- 同一 trial budget；
- 同一 invalid plan handling；
- 同一 warmup與validation成本是否計入，要明確分開報。

---

## 2.15 Result 分層

### Tier A — Confirmed

- 完整 protocol；
- validation pass；
- holdout confirmatory run；
- 95% CI；
- raw artifacts公開。

### Tier B — Exploratory

- 用於找候選；
- repetitions較少；
- 不能進 headline claims。

### Tier C — Diagnostic

- profiler開啟；
- 可解釋瓶頸，但不能當正式 latency。

### Tier D — Invalid

- OOM、crash、correctness fail、thermal drift、背景負載、資料缺失。

所有 tier 都保留，但 UI 顏色與 filter 清楚區分。

---

## 2.16 主要實驗矩陣大小控制

完整笛卡兒積會爆炸。採三階段：

### Screening

- 3 prompt buckets：128／1024／4096。
- 2 output：32／128。
- concurrency 1／4。
- 每 plan 較少 repetitions。

### Focused sweep

在每類 bottleneck 中保留 Pareto candidates，展開更多 bucket。

### Confirmatory

在 holdout prompt、real workload、不同日期 block 上重跑。

---

## 2.17 最終主表規格

主表每列必須至少包含：

```text
model
backend
format/quant
compile/cuda graph
prompt/output/concurrency
TTFT median [CI]
TPOT median [CI]
throughput [CI]
peak VRAM
startup cost
quality status
validation status
run IDs
```

若 protocol 不同，不得在相同排名欄位比較。
