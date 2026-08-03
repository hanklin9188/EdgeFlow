# 09 · UI/UX and GitHub Presentation

## 9.1 視覺目標

EdgeFlow UI應呈現：

- 安靜、可信、研究工具感；
- 不像遊戲硬體monitor；
- 不使用大量霓虹／彩虹圖；
- 數據密度高但有留白；
- measured、hypothesis、invalid一眼可分；
- 每個結論都能追溯。

設計語言：**Calm technical observatory**。

---

## 9.2 色彩

### Light mode

| Token | Hex | 用途 |
|---|---|---|
| Canvas | `#F5F7F5` | 暖灰背景 |
| Surface | `#FFFFFF` | 卡片 |
| Surface muted | `#EEF2EF` | 次層 |
| Ink | `#17212B` | 主文字 |
| Muted ink | `#62707D` | 輔助 |
| Border | `#DCE4E0` | 邊界 |
| Primary teal | `#0F766E` | measured／主要操作 |
| Indigo | `#4F46E5` | policy／secondary |
| Amber | `#B7791F` | hypothesis／warning |
| Green | `#2F855A` | pass |
| Red | `#B54747` | fail |
| Blue-gray | `#52677A` | neutral |

### Dark mode

| Token | Hex |
|---|---|
| Canvas | `#0E151B` |
| Surface | `#151F27` |
| Surface raised | `#1B2832` |
| Ink | `#E8EFEC` |
| Muted | `#9FB0B7` |
| Border | `#2B3A43` |
| Primary | `#5CC8BC` |
| Indigo | `#8D8AF7` |
| Amber | `#E4B65C` |

不得只靠顏色傳遞狀態；搭配 icon與文字。

---

## 9.3 Typography

使用系統字型，不把font files放進repo：

```css
font-family: Inter, ui-sans-serif, system-ui, -apple-system,
             "Segoe UI", "Noto Sans TC", sans-serif;
```

- Hero：48–56 px desktop。
- Page title：28–32 px。
- Card metric：26–34 px。
- Body：15–16 px，line-height 1.55–1.7。
- Monospace：`ui-monospace, SFMono-Regular, Consolas, monospace`。

避免全頁過小字；table可14 px但需sticky header。

---

## 9.4 核心頁面

### 1. Overview

顯示：

- hardware fingerprint；
- validated runs；
- active policy；
- best plan per workload profile；
- cold vs steady winner；
- recent regressions；
- quality status。

卡片不可用假的headline數字；未跑時顯示 `Awaiting validated run`。

### 2. Tune Workspace

左側：workload builder。

- model；
- prompt/output distribution；
- concurrency；
- session N；
- objective；
- quality profile；
- backend scope。

右側：candidate／Pareto／estimated experiment cost。

### 3. Run Explorer

- filter by model/backend/status/date；
- table；
- metric distribution；
- raw artifact links；
- compare checkbox。

### 4. Evidence Graph

中心顯示：

```text
Observation → Hypothesis → Intervention → Outcome → Policy rule
```

Node badge：measured/inferred/hypothesis/rejected。

### 5. Profiler View

- prefill/decode breakdown；
- kernel time share；
- gap ratio；
- launch count；
- memory timeline；
- trace download。

不要自行重畫完整Nsight timeline；提供摘要與原始trace。

### 6. Quality & Pareto

- latency vs quality；
- VRAM vs quality；
- filter strict/balanced/memory-first；
- quantization cards。

### 7. Policy

每條rule顯示：

- when；
- selected plan；
- expected metrics；
- confidence；
- evidence；
- fallback；
- last validated fingerprint。

### 8. Copilot（optional）

聊天區旁永遠顯示 evidence drawer。回答數字可點擊run。

---

## 9.5 Chart 規範

- 同一圖最多5個series；
- validated實線，exploratory虛線，invalid不連線；
- error bar預設顯示；
- Y軸不得任意截斷造成誤導；若截斷要標示；
- latency用ms，throughput用tok/s；
- cold/startup與steady分圖或清楚分面；
- 不把TTFT和TPOT加成單一未解釋score；
- tooltip顯示run ID與validation。

---

## 9.6 UI 狀態文案

### 沒資料

> No validated runs match this workload. Run a screening experiment or widen the policy scope.

### Hypothesis

> This is a profiler-supported hypothesis, not a confirmed cause. A matched intervention is required.

### Quality fail

> Faster, but outside the selected quality constraint. Hidden from automatic recommendation.

### Stale policy

> The driver or runtime fingerprint changed. Revalidation is required before deployment.

---

## 9.7 GitHub README 首屏

建議：

```text
EdgeFlow
Evidence-backed local LLM inference tuning for consumer NVIDIA GPUs.

[Dashboard screenshot]

Profile → Diagnose → Intervene → Verify → Deploy
```

Badges：

- Python；
- CUDA；
- RTX 4080 SUPER tested；
- CI；
- license；
- reproducibility；
- docs。

首屏下方四個小卡：

- Workload-conditioned policy；
- Cold + steady objective；
- Causal evidence graph；
- Correctness-gated Triton kernels。

---

## 9.8 README 必備章節

1. What EdgeFlow does。
2. Why it is different。
3. Current verified hardware。
4. Headline results（有實測後；否則TBD）。
5. 90-second demo。
6. Architecture。
7. Supported models/backends。
8. Benchmark protocol。
9. Evidence example。
10. Custom kernel。
11. Reproduce。
12. Dashboard。
13. Honest limitations。
14. Roadmap。
15. License/citation。

---

## 9.9 Results 呈現

主README只放3–5個headline；詳細放：

- `docs/RESULTS.md`；
- `docs/METHOD.md`；
- `docs/AUDIT.md`；
- `docs/KERNEL.md`；
- `data/processed/*.csv`；
- `runs/manifests/`。

每個headline附近必須有：

```text
Hardware · Model · Workload · Metric · CI · Quality · Run IDs
```

禁止：

- 把mock UI數字當result；
- 只放up-to；
- 跨不同hardware混排；
- 隱藏failed plan；
- 把estimated標成measured。

---

## 9.10 Repository 首頁視覺資產

需要：

- 1張dashboard hero screenshot；
- 1張architecture；
- 1張causal loop；
- 1張Pareto；
- 1張kernel heatmap；
- 1個15–30秒GIF：建立workload → tune →看evidence。

圖的文字保持少，細節放caption或docs。

---

## 9.11 GitHub Pages

`ui-prototype/` 可先作靜態mock；正式版由processed JSON驅動。

Pages必須明示：

- Demo data／Measured data；
- last verified date；
- environment fingerprint；
- no model inference in browser（若只是dashboard）。

---

## 9.12 Portfolio 敘事

履歷／GitHub profile：

> Built EdgeFlow, an evidence-backed LLM inference autotuner for an RTX 4080 SUPER. The system separates cold-start and steady-state cost, diagnoses runtime bottlenecks through profiler-guided interventions, validates recommendations against quality and correctness gates, and emits workload-conditioned deployment policies across PyTorch, llama.cpp, and vLLM.

Custom kernel bullet只有在完成E24後加入。

---

## 9.13 現成 UI 原型

本ZIP包含：

- `ui-prototype/index.html`
- `ui-prototype/styles.css`
- `ui-prototype/app.js`

它示範Overview、policy cards、evidence chain、run table與dark mode。所有數字標記為demo。
