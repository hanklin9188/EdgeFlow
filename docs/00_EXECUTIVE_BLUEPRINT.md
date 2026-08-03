# 00 · Executive Blueprint

## 0.1 專案名稱

**EdgeFlow**
**Causal, Workload-Conditioned Autotuning for Local LLM Inference**

中文定位：

> 一套針對消費級 NVIDIA GPU 的本機 LLM 推論最佳化系統；它依據實際 workload 量測瓶頸，透過受控制的 intervention 驗證原因，再產生兼顧延遲、吞吐、記憶體、品質與啟動成本的 deployment policy。

---

## 0.2 問題陳述

現在本機 LLM 部署面臨的問題不是「缺少推論框架」，而是框架與最佳化選項太多：

- PyTorch eager 或 `torch.compile`？
- `default`、`reduce-overhead`、`max-autotune` 哪一個值得？
- llama.cpp 的 Q8、Q6、Q5、Q4 哪個真正適合目前 workload？
- vLLM 的 batching、chunked prefill、token budget 應如何設定？
- 短 context、長 context、batch evaluation 與長時間服務是否應共用同一個 plan？
- compile、autotune、model load 的前置成本是否能在實際 session 內攤平？
- 量化造成的品質損失是否在可接受範圍？
- 某項最佳化變快，是因為消除 launch gap、降低 weight traffic，還是改變 scheduling？

現有工具通常只處理其中一層：

- compiler 在自己的 graph／kernel 空間內 autotune；
- serving runtime 在自己的 scheduler 內調整；
- profiler 提供 trace，但不替使用者完成受控驗證；
- benchmark 顯示數字，但不建立適用範圍與因果證據。

EdgeFlow 要補的是跨層決策：

\[
(h, m, w, u) \longrightarrow p^\star
\]

其中：

- \(h\)：hardware fingerprint；
- \(m\)：model architecture／size／format；
- \(w\)：prompt、output、concurrency、arrival pattern；
- \(u\)：使用週期與目標，例如短 session 或常駐服務；
- \(p^\star\)：backend、precision、quantization、compile、batching、cache 與 kernel 組成的 execution plan。

---

## 0.3 核心 Motivation

### Motivation A — 最佳設定是條件式，不是固定答案

同一張 GPU、同一模型，在不同 workload 下可能由不同 bottleneck 主導：

- batch-1 decode 可能受 weight movement 與 launch overhead 限制；
- 長 prompt prefill 更偏向矩陣乘法與 attention；
- 高 concurrency 會改變 batching 與 queueing trade-off；
- 動態 shape 可能觸發 graph specialization 或 recompilation；
- 低精度在小 batch 與大 batch 的收益可能不同。

因此，EdgeFlow 不輸出單一「冠軍設定」，而輸出條件式策略：

\[
\pi^\star(x) : x \mapsto p,
\]

其中 \(x\) 是 request／session state。

### Motivation B — steady-state winner 不一定是 deployment winner

假設 compiled plan 需要 90 秒前置成本，每次 request 只省 50 ms，break-even 為：

\[
N_{\text{break-even}}=\frac{90}{0.05}=1800.
\]

對每天只跑 20 次 request 的本機工具，它不是最佳解；對常駐服務則可能值得。

EdgeFlow 的 session objective：

\[
J_N(p)=C_{\text{load}}(p)+C_{\text{compile}}(p)+C_{\text{capture}}(p)+N\,C_{\text{request}}(p).
\]

### Motivation C — 「為什麼快」必須可驗證

EdgeFlow 將 recommendation 拆成：

```text
Observation → Hypothesis → Intervention → Outcome → Scope
```

例如：

```text
Observation: kernel gap ratio = 27%, median kernel = 8.1 μs
Hypothesis: CPU launch overhead dominates decode
Intervention: enable CUDA Graph / reduce-overhead
Outcome: gap → 10%, TPOT ↓ 14.2%
Scope: batch=1, prompt≤1024, static output bucket
```

系統不能只看相關性就宣稱原因；它必須安排 matched experiment，僅改變一個主要因素。

---

## 0.4 主要方法主張

### Claim 1 — Workload-conditioned policy

EdgeFlow policy 在實際 workload distribution \(\mathcal D\) 上，應優於任何固定 plan：

\[
\mathbb E_{x\sim\mathcal D} J(\pi_{\text{EdgeFlow}}(x),x)
<
\min_p \mathbb E_{x\sim\mathcal D}J(p,x).
\]

驗證時不能只挑 EdgeFlow 有利的 bucket；必須報全分布與 per-bucket regret。

### Claim 2 — Causal search efficiency

在接近相同最終 objective 的條件下，profiler-guided intervention 應使用更少 trials：

- grid search；
- random search；
- Bayesian／TPE baseline；
- EdgeFlow diagnose-and-intervene。

主要指標：

- trials-to-ε-best；
- tuning wall time；
- final regret；
- invalid recommendation rate；
- evidence-supported conclusion rate。

### Claim 3 — Amortization-aware deployment

相較只看 steady-state TPOT 的選擇器，EdgeFlow 應降低實際 session cost：

- 5、20、100、1000 request/session；
- cold start；
- warm restart；
- persistent service。

---

## 0.5 產品邊界

### MVP 會做

- 單張 RTX 4080 SUPER。
- Windows 11 + WSL2 或 Ubuntu。
- PyTorch eager、`torch.compile`、llama.cpp CUDA、vLLM。
- BF16／FP16、GGUF Q8/Q6/Q5/Q4；其他量化視 backend 支援。
- engine-only 與 OpenAI-compatible serving benchmark。
- profiler-driven bottleneck taxonomy。
- deterministic plan optimizer。
- quality gate。
- static Web dashboard。
- 一個真正整合到 runtime dispatch 的 Triton optimization。

### MVP 不做

- 不重寫完整 LLM serving engine。
- 不做多 GPU／distributed inference。
- 不把 Agent 當核心決策器。
- 不訓練新的 foundation model。
- 不宣稱從 RTX 4080 SUPER 直接泛化到 H100／B200。
- 不把不同 prompt、sampling、tokenizer 或 quality protocol 的數字放在同一公平比較表。
- 不將 profiler 開啟時的 latency 當成正式性能數字。

---

## 0.6 成功定義

### 工程成功

- 任一正式 run 都能由 `run_manifest.json` 重現。
- 所有結果能追溯到 environment、model hash、runtime commit、workload seed。
- backend failure 不污染其他 run。
- correctness failure 的 plan 永遠不進入 ranking。
- UI 中每個數字都能點回 raw artifact。

### 方法成功

- 至少在兩個模型族、三種 workload profile 上，conditioned policy 的 expected objective 優於最佳固定 plan。
- 至少三項 intervention 能以對照實驗支持或否定 bottleneck hypothesis。
- 至少一項 custom kernel 在完整 shape sweep 中有清楚的 faster region、fallback region 與 end-to-end benefit。
- learned component 即使加入，也只做 candidate pruning；最終 selection 仍由實測確認。

### 作品集成功

訪客在 90 秒內可以回答：

1. EdgeFlow 解決什麼問題？
2. 它與一般 benchmark wrapper 有何不同？
3. RTX 4080 SUPER 上得到什麼可驗證結果？
4. 哪些結果是 measured，哪些只是 hypothesis？
5. 如何一鍵重現？
6. 作者真正寫了哪些系統元件與 kernel？

---

## 0.7 完成後的主敘事

```text
Real workload distribution
        ↓
Cross-runtime measurement
        ↓
Profiler-grounded bottleneck diagnosis
        ↓
Controlled intervention
        ↓
Evidence-backed policy synthesis
        ↓
Local deployment recommendation
```

README 的一句話應保持：

> EdgeFlow turns local LLM deployment from ad-hoc benchmark chasing into an evidence-backed, workload-conditioned optimization process.
