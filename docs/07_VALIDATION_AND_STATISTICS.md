# 07 · Validation and Statistics

## 7.1 Validation 是產品核心

EdgeFlow 的 recommendation 若沒有 validation，只是自動化 benchmark chasing。正式 run 的 gate 順序固定：

```text
G0 Schema
G1 Environment
G2 Functional correctness
G3 Timing integrity
G4 Stability / thermal
G5 Statistical sufficiency
G6 Quality
G7 Provenance
G8 Policy eligibility
```

每一 gate 都輸出 machine-readable check，不接受只在 log 裡寫一句「看起來正常」。

---

## 7.2 Verdict

- `PASS`：可進 confirmatory table與policy。
- `CONDITIONAL_PASS`：可作探索或特定scope，不能當通用winner。
- `FAIL`：功能或品質不符合；不進ranking。
- `INVALID`：量測本身無效，例如背景GPU負載、資料遺失、timer錯誤。
- `SKIPPED`：capability不支援；不是失敗結果。

---

## 7.3 Gate 詳細規則

### G0 Schema

- manifest／plan／workload／metrics符合schema；
- required fields完整；
- ID/hash一致；
- timestamp合理；
- raw artifact存在。

### G1 Environment

- GPU與期望fingerprint一致；
- driver/CUDA/backend version符合experiment block；
- git dirty state已記錄；
- 無未允許背景GPU process；
- disk空間足夠；
- model hash正確；
- power limit／clock policy記錄。

### G2 Functional correctness

#### 同精度

- reference可執行；
- no NaN/Inf；
- next-token top-1 agreement；
- logit tolerance；
- generated token sequence（greedy）一致或差異被分析。

#### Custom kernel

- dtype/shape sweep；
- `torch.testing.assert_close`；
- unsupported shape fallback；
- race/determinism；
- repeated seed。

#### Quantized

不要求與BF16逐token一致；進quality gate比較。

### G3 Timing integrity

- warmup與measurement分離；
- CUDA sync/event正確；
- first compile另記；
- output token count符合；
- profiler未開啟正式run；
- timer monotonic；
- no missing timestamps；
- engine/e2e boundary標記。

### G4 Stability

預設：

- request latency CV或robust CV在規定內；
- first/last third median drift ≤ 3%；
- matched pair temp差 ≤ 5°C；
- GPU clock沒有長時間異常下降；
- background utilization低；
- timeout／retry率低於門檻。

不穩定時先增加repetition、修環境，不可用平均值掩蓋。

### G5 Statistical sufficiency

- repetitions達protocol；
- paired prompts完整；
- CI可計算；
- practical effect threshold；
- confirmatory與exploratory分開；
- search data與holdout不重疊。

### G6 Quality

依profile：strict/balanced/memory-first。

任何 quality fail 的plan可保留在Pareto explorer，但UI預設不推薦。

### G7 Provenance

- model/dataset revision；
- runtime commit；
- config hash；
- commands；
- random seed；
- artifact checksums；
- license note。

### G8 Policy eligibility

- PASS或明確scope的conditional；
- supporting runs ≥ minimum count；
- holdout confirm；
- software fingerprint仍valid；
- fallback可用；
- evidence record完整。

---

## 7.4 Outlier Policy

不能看到慢run就刪除。

允許排除條件必須預先定義：

- process crash；
- OS suspend；
- background GPU process出現；
- model reload意外發生；
- request timeout；
- instrumentation error。

純粹 latency高不是排除理由。保留原始值與排除reason。

---

## 7.5 Bootstrap

### Paired median difference

對每個paired prompt：

\[
d_i = y_{A,i}-y_{B,i}.
\]

bootstrap resample pairs 10,000次，計算median difference與95% percentile CI。

### Speedup

每pair ratio可能不穩定，建議同時報：

- ratio of medians；
- paired log-ratio geometric mean；
- bootstrap CI。

### Tail latency

p95需要足夠request數。online serving至少200，較穩定建議1,000。少量樣本不得過度解讀p99。

---

## 7.6 Regression Detection

每個release執行canary：

- smoke model；
- 3 workload buckets；
- 2 backends；
- custom kernel shapes。

Regression threshold：

- correctness任何下降立即fail；
- median latency >5%且CI不重疊，警告／fail；
- VRAM >5%或OOM boundary下降，警告；
- compile time >15%，警告。

需保存歷史hardware fingerprint，避免driver change誤判code regression。

---

## 7.7 Quality Statistics

### Accuracy

報 exact count、accuracy、Wilson 95% CI；paired teacher/backend comparison可用McNemar test，但主要仍報差異與CI。

### Perplexity

同模型同tokenizer比較；報token count與window protocol。PPL ratio：

\[
r_{ppl}=\frac{PPL_p}{PPL_{reference}}.
\]

### Code

pass@1固定單一sample；sandbox error與model wrong分開。

---

## 7.8 Evidence Strength

| Level | 條件 |
|---|---|
| E0 | 未驗證想法 |
| E1 | 單次觀察 |
| E2 | 重複量測相關性 |
| E3 | matched intervention + mediator |
| E4 | holdout confirm |
| E5 | 跨模型／日期／硬體 confirm |

RTX 4080 SUPER MVP的headline因果claim至少E4；不能冒充E5。

---

## 7.9 Report Wording

允許：

- “On the validated RTX 4080 SUPER setup…”
- “The intervention reduced kernel-gap ratio and TPOT under batch-1 decode…”
- “This supports a launch-overhead explanation within the tested scope.”

避免：

- “This proves CUDA Graph is always faster.”
- “Q4 is the best quantization.”
- “EdgeFlow generalizes to all GPUs.”
- “Up to X×” without distribution與scope。

---

## 7.10 詳細操作

機器可執行流程寫在：

[`../skills/edgeflow-validation/SKILL.md`](../skills/edgeflow-validation/SKILL.md)
