# 12 · Risks and Architectural Decisions

## 12.1 Risk Register

| ID | 風險 | 可能性 | 影響 | 緩解 |
|---|---|---:|---:|---|
| R1 | WSL2 profiler counters不可用 | 中 | 中 | capability probe；native/CLI fallback；不猜metric |
| R2 | 4080S 16GB限制模型 | 高 | 中 | 3–4B BF16主線；8B quantized |
| R3 | 新模型backend支援不一致 | 高 | 中 | primary選官方BF16+GGUF；support matrix |
| R4 | `torch.compile` graph break | 高 | 中 | diagnostic fullgraph；partial compile；記錄 |
| R5 | benchmark variance | 中 | 高 | thermal block、paired randomization、CI |
| R6 | 跨runtime不公平 | 高 | 高 | E09 fairness audit |
| R7 | custom kernel micro-only | 高 | 中 | profile-first；E24；誠實標示 |
| R8 | matrix太大 | 高 | 高 | screening→focused→confirmatory |
| R9 | cost model資料不足 | 高 | 低 | optional；rule baseline |
| R10 | Copilot編數字 | 中 | 高 | tool-only、grounding eval、可不發布 |
| R11 | dataset license／PII | 中 | 高 | sample IDs/hash；不重發gated raw |
| R12 | software drift | 高 | 中 | pin、canary、stale policy |
| R13 | fake-looking UI metrics | 中 | 高 | DEMO badge；TBD；production data separation |
| R14 | 只測一張GPU泛化弱 | 高 | 中 | scope清楚；後續external hardware |
| R15 | vLLM consumer GPU OOM | 中 | 中 | smaller model、memory utilization sweep |
| R16 | model vision component增加干擾 | 中 | 中 | text-only loading audit；secondary Llama |

---

## 12.2 ADR-001 — 核心不使用 Agent

**Decision**：Agent只作optional control/presentation layer。

**理由**：

- performance數字必須deterministic；
- 避免GPU資源污染；
- 避免作品定位變成普通Agent；
- 核心在沒有LLM服務時仍完整可用。

---

## 12.3 ADR-002 — 第一硬體鎖定 RTX 4080 SUPER

**Decision**：先深入單硬體，不假裝泛化。

**理由**：

- 使用者真實可持續測量；
- 能做大量repeat與profile；
- 消費級16GB場景清楚；
- 後續再做transfer。

---

## 12.4 ADR-003 — 先用官方BF16+GGUF模型

**Decision**：Ministral 3 3B作primary cross-runtime。

**理由**：減少權重來源與conversion差異；Apache-2.0；適合edge。

**風險**：multimodal component。以Llama 3.2 3B作第二族，並做text-only fairness audit。

---

## 12.5 ADR-004 — 最終winner一定實測

Learned cost model只能prune/order candidates。任何policy leaf都需要measured support。

---

## 12.6 ADR-005 — Cold與steady同等重要

不把compile/load cost藏在附錄；session objective是一級method。

---

## 12.7 ADR-006 — Profiler數字不當正式latency

Profiler用於diagnosis；正式claim用unprofiled run。

---

## 12.8 ADR-007 — Custom kernel profile-first

第一個kernel在E21後決定。規格提供候選但不預先承諾某個一定有收益。

---

## 12.9 ADR-008 — Quality是hard gate

量化plan即使最快，只要不符合使用者quality profile就不能自動推薦。

---

## 12.10 ADR-009 — UI所有數字可追溯

任何card/chart data都需run ID、validation與source type。Mock資料在production build明顯隔離。
