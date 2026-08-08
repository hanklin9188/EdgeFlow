# EdgeFlow 整體進度總覽

更新日期：2026-08-09

對應分支：`agent/formal-experiments-ui-polish`

對應 commit：`cd99676`

本文依據當前 worktree、已封存 experiment artifacts、`audit_formal_readiness.py`
與 `audit_learned_layer.py` 的重新審計結果整理。

> 進度判定原則：「有程式碼」不等於「正式實驗結論已成立」。只有通過預註冊
> protocol、quality/correctness gate、G0–G8 與所需 holdout 的結果，才列為正式完成。

## 一頁結論

- **Local-first Web App 工程主體已完成**：程式、模型、GPU、SQLite、artifacts
  與 managed runtimes 都在本機執行；瀏覽器只是 `127.0.0.1` 的操作介面，不是純雲端網站。
- **14 項實驗已在明確 scope 內完成正式驗證**：E00–E03、E05–E06、E09、
  E19–E24、E28。
- **3 項已完成執行或 primary scope，但不能外推為完整 catalog 結論**：
  E04、E07、E08。
- **E10 是目前最大的 GPU 長時間任務**：Plan A 已 45/45，Plan B 已
  23/45，兩個 plan 的共同網格為 **23/45（51.1%）**，尚差 22 buckets。
- **E11–E18 的正式 controlled intervention evidence 尚未建立**；其中部分
  runner/schema 已有，但不能當成實驗通過。
- **E25–E27 是前置條件阻擋，不是程式故障**：E25 目前 80/2,000
  unique validated points；E27 目前 0/50 per available class。
- **E29–E30 需外部資源**：E29 需至少 5 位同領域使用者，E30 需第二台
  乾淨機器，無法只在目前這台 RTX 4080 SUPER 上自我宣告完成。

## 狀態定義

| 標記 | 意義 |
|---|---|
| ✅ 正式完成 | 已有可追溯 artifact，且在文件所列的 exact scope 內通過 |
| 🟡 部分完成 | 工程可執行或已有部分正式 evidence，但 catalog 要求尚未全部滿足 |
| ⛔ 前置阻擋 | 程式會主動拒絕訓練或宣告結論，直到資料量或 labels 達標 |
| ⬜ 外部驗證 | 需使用者、第二台機器或外部環境 |

## E00–E30 逐項進度

| ID | 目標 | 目前狀態 | 已完成的證據 | 尚未完成／限制 |
|---|---|---|---|---|
| E00 | 硬體／軟體 fingerprint | ✅ | RTX 4080 SUPER 正式 artifact `PASS` | 無 |
| E01 | timer calibration | ✅ | 正式 artifact `PASS` | 無 |
| E02 | thermal/background noise | ✅ | 每 condition 1,000 次、30 秒 stabilization，7/7 gates `PASS` | 只支持已量測機器與環境 |
| E03 | repetition/confidence stability | ✅ | 正式 artifact `PASS` | 無 |
| E04 | PyTorch eager baseline | 🟡 | Llama 3.2 3B 正式 12-case matrix 全數執行，0 failures；1 case policy-eligible | 11 cases 為 G4 `CONDITIONAL_PASS`，不能宣告整個 eager matrix 的廣義性能結論 |
| E05 | `torch.compile` modes | ✅ | 32-case matrix 完成，0 failures；confirmatory case `PASS` | 16 個不相容 CUDAGraph 組合依能力規則剪枝，結論限於可支援 scope |
| E06 | dynamic-shape recompilation | ✅ | mixed-shape formal evidence `PASS` | 結論限於已量測 sequence/model |
| E07 | llama.cpp quantization | 🟡 | Primary Ministral 3B Q5_K_M quality + `p1024-o128-c1` 正式 run 通過；Q4_K_M 保留為 quality negative evidence | Q8_0、Q6_K 與完整 workload sweep 尚未全部成立 |
| E08 | vLLM scheduler | 🟡 | Primary Ministral 3B BF16 quality + graph `p1024-o128-c1` 正式 run 通過 | 完整 scheduler parameter sweep 尚未全部成立 |
| E09 | cross-runtime fairness audit | ✅ | Ministral exact-scope audit `PASS`，兩邊均 30 repetitions | llama.cpp Q5_K_M eager 與 vLLM BF16 graph 同時差異，因此 `causal_runtime_isolation=false`；只能做 quality-gated 描述性比較 |
| E10 | fixed-plan dominance | 🟡 | Plan A 45/45 `PASS`；Plan B 23/45；common grid 23/45 | Plan B 尚差 22 buckets；審計狀態仍是 `INCOMPLETE`，目前 oracle gain 0.309% 只是 partial-grid 描述，不得當正式 policy 結論 |
| E11 | policy synthesis baselines | 🟡 | 有部分工程支援 | 需依賴完整 E10 與正式 baseline 對照 |
| E12 | real-distribution replay | 🟡 | catalog/protocol 已定義 | 未建立真實分佈 replay dataset 與正式 artifact |
| E13 | launch-overhead intervention | 🟡 | intervention schema/runner 部分實作 | 缺 controlled mediator、negative control 與 holdout pairs |
| E14 | memory-bandwidth intervention | 🟡 | intervention schema/runner 部分實作 | 缺正式 controlled intervention evidence |
| E15 | compute-bound prefill intervention | 🟡 | intervention schema/runner 部分實作 | 缺正式 controlled intervention evidence |
| E16 | KV-cache capacity/long context | 🟡 | protocol 已定義 | 容量邊界與 long-context 正式實驗未執行 |
| E17 | scheduler-bound mixed workload | 🟡 | heterogeneous workload 工程支援部分完成 | 缺正式 intervention/holdout evidence |
| E18 | negative-control diagnosis | 🟡 | deterministic diagnosis 與 audit 工程已實作 | 尚無 E11–E17 對應的正式 negative-control evidence |
| E19 | session break-even | ✅ | exact-scope artifact `PASS` | 不外推至未量測 plan/workload |
| E20 | cold-start/warm restart | ✅ | 30 對 fresh-process/cached-host 實驗 `PASS` | 真實 OS cache drop、machine reboot、external service restart 未驗證 |
| E21 | profile-first candidate selection | ✅ | 10 次累積 CUDA profile iterations，`PASS`；下一候選為 `fused-rope-v1` | 不能反向用來證明既有 RMSNorm E24 的選擇是 profile-first |
| E22 | Triton correctness | ✅ | 216/216 correctness checks 通過 | 只支持列出的 dtype/shape scope |
| E23 | Triton performance sweep | ✅ | 216-row randomized sweep 的 correctness/fallback 規則通過 | 性能主張仍受 E24 exact scope 限制 |
| E24 | end-to-end kernel integration | ✅ | Llama 3.2 3B BF16，30+30 paired search/untouched holdout，`END_TO_END_SUPPORTED` | 不外推到其他模型、runtime 或 shape |
| E25 | cost-model dataset | ⛔ | exporter/dedup/split/audit 已實作；80 unique validated points | 門檻為 2,000，尚差 1,920；重複 runs/repetitions 不能灌水 |
| E26 | cost-model evaluation | ⛔ | grouped + temporal holdout protocol 已實作 | E25 未達 2,000 前拒絕訓練與主張 |
| E27 | bottleneck classifier | ⛔ | label exporter/audit 已實作 | 目前 0 unique intervention labels；每個 available class 需 50 |
| E28 | grounded copilot | ✅ | 固定 22 題，22/22 citation/refusal checks 通過，unsupported numeric claims = 0 | 回答 scope 仍僅限當前 artifacts 可支持的內容 |
| E29 | UI usability audit | ⬜ | UI 工程、responsive layout 與實機流程已建立 | 需至少 5 位同領域使用者完成 usability tasks |
| E30 | reproducibility challenge | ⬜ | package/manifest/validation/CI 已建立 | 需第二台機器或真正乾淨環境重現 |

## 工程與 UI 進度

### 已完成

- Local-first API、SQLite artifact index、typed worker queue、CLI 與 localhost-only bind。
- llama.cpp / vLLM managed runtime 的 start → authenticated inference → stop 流程。
- Host、Origin、session token、request body、path 與 shell boundary 防護。
- 正式 dashboard 只讀真實 SQLite/artifacts，不把 mock 數字當正式 evidence。
- 桌面寬版與窄版 responsive UI、較明亮的配色、進度與實驗狀態呈現。
- runner 已有 low-overhead NVML telemetry、thermal warmup/inter-bucket idle barrier、
  concurrent measurement 與 sequential correctness separation、server-reported completion-token accounting。

### 不是缺失項目

- 目前選定的產品形態是 **Local-first Web App**，因此看到 `http://127.0.0.1:<port>`
  是預期行為。Native Windows `.exe` 不在目前已選定的主線交付範圍。

## 目前主要差距

1. **E10 完整 common grid**：續跑 Plan B 剩餘 22 buckets，完成後重跑
   formal-readiness audit，才能判定 fixed plan 或 conditioned policy 是否真的有意義。
2. **E11–E18 causal chain**：建立 real-distribution replay，再依 launch、memory、
   compute、KV-cache、scheduler bottleneck 執行 controlled intervention、mediator、
   negative control 與 holdout pairs。
3. **E25–E27 data prerequisites**：只能透過新的、互異的硬體-plan-workload
   正式點數與 intervention labels 累積，不能拿 repetitions 或重跑當新樣本。
4. **E29–E30 external validation**：需實際參與者與第二台乾淨機器。

## 建議後續順序

1. 從 `ministral3-3b-grid-graph-mbt65536` checkpoint 續跑 E10 剩餘 22 buckets。
2. 重建 E10 formal-readiness，將 45/45 common grid 結果封存。
3. 先完成 E12 real-distribution replay，再依診斷順序執行 E13–E18。
4. 用新 evidence 重跑 E25–E27 prerequisite audit；達標前不啟動 learned model 訓練。
5. 邀請領域使用者完成 E29，並在第二台機器完成 E30。

## GitHub 與可追溯狀態

- Repository：`hanklin9188/EdgeFlow`
- Branch：`agent/formal-experiments-ui-polish`
- Draft PR：`#3`
- 本次整理前 commit：`cd99676`
- 當前未有 E10/vLLM/llama.cpp 實驗進程在背景執行；Plan B 的
  `RUNNING` 代表 checkpoint 可續跑，不代表當前仍有 process 在運作。

## 完整完成的判定

要宣告整體規劃完成，至少還要同時滿足：E10 45/45 雙 plan common grid、
E11–E18 formal controlled evidence、E25–E27 的合法資料與 holdout 門檻、E29 人類
usability audit，以及 E30 第二環境重現。在這些條件達成前，專案是
「工程主體可用、多個 exact-scope 正式結論已成立，但完整研究驗證尚未結案」。
