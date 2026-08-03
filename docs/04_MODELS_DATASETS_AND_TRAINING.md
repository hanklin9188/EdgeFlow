# 04 · Models, Datasets, Quantization, and Training

## 4.1 最重要的結論

### MVP 不需要訓練任何 LLM

EdgeFlow 的核心價值來自：

- 量測；
- profiling；
- correctness validation；
- causal intervention；
- deterministic plan search；
- policy synthesis。

因此第一個公開版本不應把時間花在微調 foundation model。模型在這裡是**受測系統**，不是要被訓練的主體。

只有兩個後期元件可能需要訓練：

1. **Cost model**：預測某個 plan 在特定 workload 的 latency／memory，用於剪枝。
2. **Bottleneck classifier**：從 trace summary 排序瓶頸，用於選下一項 intervention。

它們都直接用 EdgeFlow 自己的 run database 訓練，不需要蒐集私人對話，也不需要重新訓練 LLM。

Performance Copilot 原則上使用現成 instruct model，不做 fine-tuning。

---

## 4.2 模型選擇原則

模型組合必須同時覆蓋：

- 可在 16 GB VRAM 上跑 BF16 的小模型；
- 有官方或可重現 GGUF 的模型；
- 至少兩個不同 architecture family；
- 一個現代 reasoning／thinking 模型；
- 一個 8B quantized memory-pressure case；
- license 與 gating 被清楚記錄。

不能只選對某個 backend 特別友善的模型。

---

## 4.3 建議模型矩陣

### M0 — Smoke model

**建議**：`meta-llama/Llama-3.2-1B-Instruct` 或一個同級、無 gating 的 0.5–1.5B 模型。

用途：

- CI smoke；
- runtime adapter 開發；
- 快速 correctness；
- 不進主要性能 claim。

若不希望依賴 Llama gated access，可在 repo 實作時改用當時官方支援良好的 Qwen／Gemma 小模型；registry 必須記錄替換理由。

### M1 — Primary cross-runtime model

**推薦**：

- HF：`mistralai/Ministral-3-3B-Instruct-2512-BF16`
- GGUF：`mistralai/Ministral-3-3B-Instruct-2512-GGUF`

理由：

- 官方 BF16 與官方 GGUF；
- Apache-2.0；
- edge-oriented；
- 3B language model 等級適合 16 GB GPU；
- 可在 PyTorch/vLLM 與 llama.cpp 間建立比較。

注意：模型包含 vision component。EdgeFlow 第一版只測 text-only；run manifest 必須記錄 vision encoder 是否載入、是否影響 VRAM。若 backend 對 text-only loading 行為不一致，需在公平性 audit 中說明。

### M2 — Familiar dense decoder family

**推薦**：`meta-llama/Llama-3.2-3B-Instruct`

用途：

- 與現有 FAD／StateFT 背景一致，但本專案暫不整合 FAD；
- 驗證方法不是只對 Mistral family 有效；
- PyTorch、vLLM baseline；
- llama.cpp 使用由原始權重在固定 commit 上自行轉換的 GGUF，避免依賴未知第三方 quantization。

限制：Llama 3.2 license 與 gated access；README 必須說明使用者自行接受條款。

### M3 — Modern reasoning-aware stress model

**推薦**：`Qwen/Qwen3.5-4B`

主要 protocol：

- text-only；
- thinking disabled；
- greedy；
- fixed output length。

另設 secondary protocol：thinking enabled，測量 reasoning token 使 workload 分布如何變化。

用途：

- 測新架構對 Transformers/vLLM/compile 的支援；
- 驗證 dynamic output 與 reasoning mode；
- 作為 optional Copilot model 候選。

注意：若 pinned llama.cpp commit 尚未支援該 architecture，標記 `unsupported`，不能用非對等權重硬湊跨 runtime 比較。

### M4 — Quantized memory-pressure model

**推薦**：`mistralai/Ministral-3-8B-Instruct-2512-GGUF`

格式：Q8_0、Q6_K、Q5_K_M、Q4_K_M。

用途：

- 16 GB VRAM capacity；
- quantization Pareto；
- 長 context與concurrency；
- llama.cpp CUDA。

FP8 vLLM／Transformers 僅在 RTX 4080 SUPER 與當前 backend 經 capability probe 及 correctness 驗證後加入，不把模型卡的「可放入 12GB」直接當成這台卡的支援保證。

---

## 4.4 模型下載與 provenance

正式 repository 不重新散佈模型權重。每個模型 entry 必須包含：

```yaml
model_id: ministral3-3b-instruct-2512
source:
  hf_repo: mistralai/Ministral-3-3B-Instruct-2512-BF16
  revision: <pinned commit>
license: apache-2.0
files:
  - name: model-00001-of-*.safetensors
    sha256: <computed locally>
tokenizer:
  revision: <pinned commit>
chat_template_sha256: <hash>
```

GGUF：

- 優先官方 GGUF；
- 否則從已 pin 的 original weight 用已 pin llama.cpp converter 產生；
- quantization command、converter commit、output hash 全部保存；
- 不混用不同模型 revision 的 BF16 與 GGUF。

---

## 4.5 Dataset 分層

### D0 — EdgeFlow Synthetic Workloads（必要）

這不是訓練資料，而是 controlled benchmark generator。

組成：

- 公開、無敏感內容的短句庫；
- deterministic seed；
- tokenizer-aware token packing；
- prompt bucket；
- output bucket；
- arrival trace；
- repeated-token special case。

公開內容：生成程式、seed、source text license、token IDs hash。

用途：

- shape controlled benchmark；
- compile／recompilation；
- kernel；
- causal matched experiment。

### D1 — UltraChat 200k（主要 real-distribution）

ID：`HuggingFaceH4/ultrachat_200k`

用途：

- prompt／turn length distribution；
- realistic chat prompt replay；
- 不用來訓練 EdgeFlow 模型。

建議 split：`test_sft`。

採樣：

1. 下載 dataset 指定 revision。
2. 對每個 target model tokenizer 計算 user prompt token length。
3. 分桶：0–128、129–512、513–1024、1025–2048、2049–4096、4097+。
4. 每桶固定 seed 抽 100–500 prompts。
5. 保存 row ID／hash，不在 repo 重發完整內容。

### D2 — LMSYS-Chat-1M（可選 gated）

ID：`lmsys/lmsys-chat-1m`

用途：更接近真實使用者對話的 workload distribution。

限制：需要接受 dataset agreement；可能包含 unsafe content；不將原文放入公開 artifacts。

EdgeFlow 必須能在沒有 D2 的情況下完整重現核心結果。D2 只作 external-validity extension。

### D3 — WikiText-2（必要 quality gate）

ID：`Salesforce/wikitext`，config `wikitext-2-raw-v1`。

用途：

- perplexity；
- quantization quality；
- backend logits consistency。

程序：

- 使用 test split；
- 固定 sliding window與stride；
- 以同一 tokenizer／BOS policy；
- 記錄有效 token 數；
- 不同 tokenizer 的 PPL 不能跨模型比較，只比較同模型不同 plan。

### D4 — ARC-Challenge（必要 quality gate）

ID：`allenai/ai2_arc`，config `ARC-Challenge`。

用途：

- multiple-choice accuracy；
- 量化與 backend quality retention。

Protocol：

- candidate conditional log-likelihood；
- fixed prompt template；
- candidate token sum；
- 是否 length normalization 必須固定並記錄；
- primary 用 full test；screening 可用固定 200-item subset。

### D5 — GSM8K（extended quality）

ID：`openai/gsm8k`，config `main`。

用途：

- reasoning output；
- thinking on/off workload；
- quantization是否影響多步推理。

Protocol：

- test split；
- greedy；
- fixed prompt format；
- 擷取 `####` 或 final numeric answer；
- 報 exact match；
- 另報 generated token length distribution。

它不是主要 latency controlled workload，因為模型可能產生不同長度；作 quality／realistic generation extension。

### D6 — HumanEval（選配）

ID：`openai/openai_humaneval`。

用途：

- code-generation request shape；
- pass@1 quality；
- 長 output。

安全執行：generated code 必須在 sandbox／container、timeout、resource limit 下執行。若只做 latency，可不執行 generated code，但不能宣稱 pass@1。

---

## 4.6 Dataset 使用表

| Dataset | Controlled latency | Real distribution | Quality | Training |
|---|---:|---:|---:|---:|
| EdgeFlow Synthetic | ✓ | — | — | — |
| UltraChat 200k | △ | ✓ | — | — |
| LMSYS-Chat-1M | △ | ✓ | — | — |
| WikiText-2 | — | — | ✓ | — |
| ARC-Challenge | — | — | ✓ | — |
| GSM8K | △ | ✓ | ✓ | — |
| HumanEval | △ | ✓ | ✓ | — |
| EdgeFlow run DB | — | — | — | optional cost model/classifier |

---

## 4.7 Quality subset 與完整評估策略

為控制開發成本：

### Screening gate

- WikiText-2 固定 10–20% token subset；
- ARC-C 固定 200 items；
- 20 prompts greedy token agreement。

### Confirmatory gate

- WikiText-2 full test；
- ARC-C full test；
- GSM8K full或預先宣告 subset；
- HumanEval optional full 164。

Screening 通過不代表可以進 headline quality claim；headline 必須用 confirmatory。

---

# Optional Training A · Learned Cost Model

## 4.8 目的

Cost model 不取代 benchmark。它預測：

\[
\hat y=f(h,m,w,p),
\]

其中 target 可為 TTFT、TPOT、VRAM、throughput與failure probability。

用途：

- 先刪除明顯差或高風險 plans；
- 排序下一批量測；
- 降低 tuning trial。

最終 plan 必須實測並通過 validation。

## 4.9 訓練資料

來源：`validation_status ∈ {PASS, CONDITIONAL_PASS}` 的 run aggregate。

一列代表一個 unique：

```text
hardware × model revision × backend plan × workload bucket × session mode
```

不要把 30 repetitions 當 30 個獨立 training rows；它們是同一 configuration 的分布。

最低建議：

- 2,000 unique points 才開始正式比較；
- 每 backend ≥ 200；
- 每主要 workload bucket ≥ 100；
- failure rows另建 binary target。

## 4.10 Features

### Hardware

- compute capability；
- VRAM；
- measured memory bandwidth proxy；
- power limit；
- CPU cores；
- WSL/native。

單 GPU MVP 中部分為常數，但 schema 保留未來擴充。

### Model

- params；
- layers；
- hidden size；
- intermediate size；
- attention heads／KV heads；
- vocab；
- architecture one-hot；
- model bytes；
- context limit。

### Workload

- prompt tokens；
- output tokens；
- batch；
- concurrency；
- request rate；
- arrival type；
- expected session N。

### Plan

- backend；
- dtype／quant；
- compile；
- CUDA Graph；
- token budget；
- KV dtype；
- flash attention；
- custom kernels。

### Diagnostic optional

- kernel gap；
- top kernel share；
- DRAM proxy；
- compile count。

這些 features 只有在 exploratory refinement 可用；初始 prediction 不能依賴尚未跑 profiler 才知道的資訊。

## 4.11 Model 選擇

由簡到難：

1. analytical memory model；
2. linear／ridge；
3. k-nearest neighbor；
4. random forest／ExtraTrees；
5. histogram gradient boosting；
6. XGBoost/LightGBM（可選 dependency）。

不建議一開始用神經網路，資料量與硬體種類不足。

## 4.12 Split

禁止 random row split。使用：

- GroupKFold by workload bucket；
- leave-one-prompt-range-out；
- leave-one-model-family-out（研究 transfer 時）；
- time-based split 測 software drift。

## 4.13 Metrics

- MAPE／SMAPE；
- Spearman ranking；
- top-3 recall；
- top-1 selection regret；
- memory constraint violation；
- failure calibration；
- trials saved。

部署接受條件重點不是回歸誤差最低，而是：

```text
prune 後仍保留真正 top-k plan，且不增加 constraint violation。
```

---

# Optional Training B · Bottleneck Classifier

## 4.14 Label 來源

禁止只用人看 trace 隨意標籤。Label 由 controlled intervention 產生：

- launch intervention 同時降低 gap 與 latency → supported launch label；
- quantization降低bytes proxy與decode latency → memory label；
- prefill matmul optimization改善長prompt → compute label；
- token budget改變 queue/ITL → scheduler label；
- 無足夠中介證據 → `insufficient_evidence`。

一個 case 可 multilabel。

## 4.15 Training

- rule engine 是 baseline與fallback；
- gradient boosted classifier；
- class weighting；
- probability calibration；
- group split by experiment block；
- 每類至少 50 supported cases才訓練；不足則不啟用。

Metrics：macro-F1、Brier score、top-2 recall、unsupported false-positive rate。

---

# Optional Copilot Model

## 4.16 是否訓練

**不訓練。** 先使用現成小型 instruct model，例如 Qwen3.5 4B，thinking disabled，或外部 API。

Copilot 只在 benchmark 完成後讀取結構化 summary。不能與受測模型同時佔用 GPU；本機模式可在 benchmark service 停止後載入，或使用 CPU／量化版本。

若未來要 fine-tune：

- 使用完全 synthetic tool-call records；
- input 為 schema-valid run summary；
- output 為 tool call／grounded report；
- 不用私人 prompt；
- 必須保留數字 fidelity 測試；
- 這不是 MVP requirement。

---

## 4.17 資料與模型公開邊界

公開 repository 應包含：

- dataset loader與處理腳本；
- dataset revision；
- sample IDs／hash；
- synthetic prompts；
- model registry；
- conversion commands；
- checksums；
- quality summary。

不包含：

- gated model weights；
-完整 LMSYS conversations；
-可能含個資的 raw prompts；
-未授權 GGUF；
-私有 API key／Hugging Face token。
