# 05 · Autotuning and Causal Method

## 5.1 問題形式化

定義系統狀態：

\[
x=(h,m,w,u),
\]

其中 hardware \(h\)、model \(m\)、workload \(w\)、usage horizon \(u\)。

Execution plan：

\[
p=(b,q,c,s,k),
\]

其中：

- \(b\)：backend；
- \(q\)：precision／quantization／KV dtype；
- \(c\)：compile、CUDA Graph、dynamic shape；
- \(s\)：batching／scheduler／token budget；
- \(k\)：custom kernel set。

EdgeFlow 尋找 policy：

\[
\pi^*=\arg\min_\pi\mathbb E_{x\sim\mathcal D}[J(\pi(x),x)]
\]

subject to：

\[
M(\pi(x),x)\le M_{\max},
\]

\[
Q(\pi(x),m)\ge Q_{\min},
\]

\[
V(\pi(x),x)=\text{PASS}.
\]

---

## 5.2 Objective

### Interactive profile

\[
J_{interactive}=
0.35\widetilde{TTFT}
+0.35\widetilde{TPOT}
+0.15\widetilde{p95\,ITL}
+0.10\widetilde{VRAM}
+0.05\widetilde{startup}.
\]

### Throughput profile

\[
J_{throughput}=
-0.55\widetilde{tokens/s}
+0.20\widetilde{p95\,latency}
+0.15\widetilde{VRAM}
+0.10\widetilde{failure}.
\]

### Session-aware profile

\[
J_N(p)=C_{startup}(p)+N\,C_{request}(p)+\lambda_M M(p)+\lambda_Q\Phi_Q(p).
\]

Quality 可採 hard constraint，不建議只用 penalty 掩蓋不合格 plan。

所有 normalized metric 需依同一 workload candidate set 計算；不能跨不相容實驗任意 normalize。

---

## 5.3 Candidate Space

### PyTorch

- eager；
- BF16／FP16；
- compile modes；
- dynamic shape；
- fullgraph diagnostic；
- CUDA Graph；
- attention implementation；
- custom kernel on/off。

### llama.cpp

- Q8_0、Q6_K、Q5_K_M、Q4_K_M；
- n_batch／n_ubatch；
- flash attention；
- KV type；
- parallel slots；
- context；
- GPU offload all。

### vLLM

- max_num_batched_tokens；
- max_num_seqs；
- GPU memory utilization；
- eager/graph；
- KV dtype；
- chunked prefill current controls；
- attention backend（capability允許時）。

Candidate generator 必須根據 backend version 的 capability report 產生，不可假設 option 永遠存在。

---

## 5.4 Search Stages

### Stage A — Static pruning

刪除：

- backend 不支援；
- model format不相容；
- estimated model+KV+workspace超過安全VRAM；
- 已知 correctness fail；
- quality profile禁止的quant；
- duplicate canonical plans。

### Stage B — Cheap screening

- 少量 bucket；
- 較少 repetitions；
- 只取核心 metric；
- 產生初始 Pareto set。

### Stage C — Diagnose

對 Pareto附近與異常 case 執行 profiler，產生 bottleneck hypothesis。

### Stage D — Intervene

每次 intervention 只改一個主要 factor；建立 matched pair。

### Stage E — Confirm

在 unprofiled、完整 repetitions、holdout workloads 重跑。

### Stage F — Synthesize policy

以簡單可解釋 rule優先；只有 rule complexity或資料量需要時才用 learned model。

---

## 5.5 Bottleneck Taxonomy

### Launch-overhead-bound

可能 evidence：

- 許多 <10–20 μs kernel；
- CPU-GPU gap高；
- SM active低；
- batch增加或CUDA Graph改善。

### Memory-bandwidth-bound

- decode階段；
- weight bytes高；
- compute利用低；
- quantization或batch增加改善；
- Nsight memory metrics支持。

### Compute-bound

- 長prefill；
- GEMM／attention占比高；
- tensor utilization高；
- max-autotune／shape alignment改善。

### KV-capacity-bound

- context/concurrency提高接近OOM；
- VRAM被KV主導；
- KV dtype或量化釋放容量。

### Scheduler-bound

- queue delay／head-of-line blocking；
- mixed prefill/decode；
- token budget改變 tail latency。

### Compile-bound

- first request dominated by compile；
- shape變化反覆recompile；
- session N不足攤平。

### I/O／load-bound

- model read、page cache、deserialization；
- warm start差異大。

### Insufficient evidence

這是必要類別。EdgeFlow 必須允許「目前無法判定」。

---

## 5.6 Causal Evidence Standard

要支持 `B → Y`，至少需要：

1. baseline observation 支持 bottleneck B；
2. intervention I 被設計來改變 B；
3. matched controls 保持其他主要因素；
4. mediator metric \(M_B\) 按預期變化；
5. outcome \(Y\) 有 practical improvement；
6. negative control 不產生相同改善，或有合理說明；
7. holdout confirm。

Evidence record：

```text
OBSERVATION
HYPOTHESIS
INTERVENTION
CONTROLLED VARIABLES
MEDIATOR RESULT
OUTCOME RESULT
STATISTICAL STATUS
SUPPORTED / REJECTED / INCONCLUSIVE
SCOPE
```

不滿足 mediator 時，只能說 intervention correlated with speedup，不能說證實 bottleneck。

---

## 5.7 Evidence Graph

Nodes：

- run；
- metric；
- observation；
- hypothesis；
- intervention；
- validation verdict；
- policy rule。

Edges：

```text
run --observes--> metric
metric --supports--> hypothesis
intervention --tests--> hypothesis
run_pair --supports/rejects--> hypothesis
hypothesis --justifies--> policy_rule
```

UI 每個 recommendation 都可沿 graph 回到 run。

---

## 5.8 Search Baseline 實作

### Grid

遍歷相同有限 candidate set。用來取得 oracle或近似oracle，但成本高。

### Random

固定 seed，uniform sampling；trial budget與EdgeFlow相同。

### TPE

輸入相同 objective；invalid plan 給固定 penalty；要計入 tuning overhead。

### Rule-only

例如：

- concurrency=1選lowest TPOT；
- concurrency≥4選vLLM；
- VRAM不足選Q4。

用來證明複雜方法是否真的有必要。

### EdgeFlow causal

由 trace分類 → intervention mapping → local neighborhood search。

---

## 5.9 Policy Representation

優先使用 decision list／tree：

```text
IF session_requests <= 20 AND concurrency = 1
    choose llama.cpp Q6
ELSE IF prompt_tokens <= 1024 AND concurrency = 1
    choose torch.compile reduce-overhead BF16
ELSE IF concurrency >= 4
    choose vLLM plan X
ELSE
    fallback validated plan
```

每個 leaf 需有最小 support count與uncertainty。資料稀疏區回退，不外推。

---

## 5.10 Drift

Runtime、driver、model revision 更新後，policy可能失效。

Drift detector：

- fingerprint change；
- canary workload latency change > 5%；
- quality mismatch；
- backend option change；
- custom kernel recompile。

Policy狀態：

```text
VALID
STALE
REVALIDATION_REQUIRED
INVALID
```

GitHub headline只引用 VALID policy。

---

## 5.11 Optional Learned Search

Cost model可用 Expected Improvement或uncertainty sampling選下一 trial，但必须遵守：

- model建議不是 evidence；
- high uncertainty／constraint boundary優先實測；
- final recommendation measured；
- 把 model inference time與training cost計入 tuning cost；
- report model ablation。

---

## 5.12 方法 Ablation

至少包含：

1. 無 workload conditioning；
2. 無 amortization；
3. 無 profiler diagnosis，純 black-box；
4. 無 mediator validation；
5. 無 quality gate；
6. 無 learned pruning；
7. full EdgeFlow。

這些 ablation 直接回答每個創新是否有實際貢獻。
