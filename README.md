# EdgeFlow

> **Causal, workload-conditioned autotuning for local LLM inference on an RTX 4080 SUPER.**

EdgeFlow turns local LLM deployment from ad-hoc benchmark chasing into an evidence-backed, workload-conditioned optimization process.

它不假設某個 runtime 或量化永遠最快。EdgeFlow 先保存逐 request 原始量測，經過 correctness、timing、stability、statistics、quality 與 provenance gate，再用 profiler observation 產生可反駁的 bottleneck hypothesis，最後只從通過驗證的 runs 建立 deployment policy。

```text
Observe → Diagnose → Intervene → Verify → Synthesize Policy
```

> [!IMPORTANT]
> Repository 內的 `examples/` 與 `ui-prototype/` 永遠標示為 `demo`，不得支持性能結論。正式產品是 localhost-only 的 Local-first Web App；模型、GPU 工作與 artifacts 不上傳雲端。目前 README 不刊登尚未完成 confirmatory/holdout 的性能數字。

## 已完成的工程面

- RTX 4080 SUPER hardware/software fingerprint 與 environment doctor。
- Pydantic + JSON Schema contracts：workload、plan、manifest、metric、profile、evidence、verdict、policy。
- PyTorch eager／`torch.compile` data plane；llama.cpp／vLLM OpenAI-compatible adapters 與 capability-safe skip。
- 精確 target-tokenizer synthetic prompt、隔離 run artifact、逐 request JSONL、SQLite index。
- G0–G8 validation engine、robust statistics、10,000-sample paired bootstrap、thermal/drift checks。
- Deterministic bottleneck diagnosis、controlled-intervention drafts、session-aware objective 與 decision-list policy。
- Correctness-cached Triton fused residual + RMSNorm，任何未驗證 shape 自動 fallback。
- Typer CLI、localhost FastAPI、隔離背景 worker、白名單 runtime service manager、Prometheus endpoint、無假數據 Local-first Web App、測試與 CI。

完整 phase 狀態與尚需正式 GPU 實驗的項目見 [Implementation Status](docs/IMPLEMENTATION_STATUS.md)。研究設計仍完整保留在 [Executive Blueprint](docs/00_EXECUTIVE_BLUEPRINT.md) 與 [Experiment Catalog](docs/03_EXPERIMENT_CATALOG.md)。

## 快速開始

環境需求：Python 3.11–3.13、NVIDIA driver、CUDA-enabled PyTorch。推薦使用 WSL2 Ubuntu 或原生 Ubuntu。

```bash
uv sync --extra dev
source .venv/bin/activate
edgeflow doctor
pytest -q
```

GPU data plane 需要現有 CUDA PyTorch/Transformers 環境，或：

```bash
uv sync --extra dev --extra gpu --extra quality
```

`quality` extra 提供固定 revision 的 WikiText-2／ARC-Challenge evaluator；控制平面本身仍可維持 lean environment。

### 1. Inspect

```bash
edgeflow inspect
edgeflow inspect --json --output artifacts/hardware_fingerprint.json
edgeflow doctor
```

### 2. Define a workload

```bash
edgeflow workload create \
  --model smollm2-360m-instruct \
  --profile local-agent \
  --prompt-distribution 32 \
  --output 8 \
  --concurrency 1 \
  --session-requests 30 \
  --save configs/generated/smoke-workload.json
```

分布格式同樣支援 `512:0.25,1024:0.45,2048:0.30`；每次實際 request 會在目標 tokenizer 下精確生成指定 token 數。

### 3. Screen capabilities

```bash
edgeflow tune screen \
  --workload configs/generated/smoke-workload.json \
  --parameter-count 360000000 \
  --save artifacts/planned_candidates.json
```

Screening 只做 capability/memory/duplicate pruning，輸出是候選，不是 recommendation。

### 4. Benchmark

```bash
edgeflow benchmark run \
  --model-ref HuggingFaceTB/SmolLM2-360M-Instruct \
  --workload configs/smoke/workload.json \
  --plan configs/smoke/pytorch-eager.json \
  --repetitions 30 \
  --warmup 5 \
  --experiment-id E04
```

每個 run 會產生：

```text
artifacts/<run_id>/
├── run_manifest.json
├── hardware_fingerprint.json
├── workload.json
├── execution_plan.json
├── metrics.jsonl
├── stdout.log
├── stderr.log
├── validation_verdict.json
└── VALIDATION.md
```

Formal policy eligibility 還要求 correctness 與 quality artifacts；單純 smoke latency 預期得到 `CONDITIONAL_PASS`，不會被包裝成完整結論。

先建立可重用、精確綁定 model revision／dtype／backend 的本機 BF16 quality reference：

```bash
python scripts/evaluate_hf_quality.py \
  --model-id llama-3.2-3b-instruct \
  --wikitext-tokens 8192 \
  --arc-samples 50 \
  --allow-download
```

後續相同 scope 的 PyTorch eager／`torch.compile` run 會自動複製該 report；quantized 或不同 runtime 不會沿用，以免跨格式外推品質。

正式 E04／E05 網格可續跑；每完成一個 case 就更新本機 matrix artifact，重新執行會略過已完成 case：

```bash
python scripts/run_pytorch_matrix.py E04 --model-id llama-3.2-3b-instruct
python scripts/run_pytorch_matrix.py E05 --model-id llama-3.2-3b-instruct
```

正式模式會拒絕 dirty checkout 與缺少 exact-scope quality report 的模型；`--quick` 只用於工程 regression，永遠標為 `DEVELOPMENT`。

### 5. Validate and diagnose

```bash
edgeflow validate artifacts/<run_id>
edgeflow profile --run <run_id> --level nsys
edgeflow diagnose --profile examples/sample_profiler_summary.json
python scripts/verify_results.py
```

`profile` 會產生隔離 diagnostic rerun 命令。Profiler-enabled latency 永不覆寫 unprofiled production timing。

### 6. Validate the Triton path

```bash
edgeflow kernel validate
edgeflow kernel validate --full
python scripts/benchmark_rmsnorm.py --quick
```

Dispatcher key 包含 kernel version、GPU、dtype 與 shape；cache 中沒有 `PASS` 時一定使用 PyTorch reference。

### 7. Start the Local-first Web App

選用 llama.cpp／vLLM 前，先建立互不污染的固定版本環境：

```bash
./scripts/bootstrap_llama_cpp.sh
./scripts/bootstrap_vllm.sh
```

```bash
edgeflow serve --host 127.0.0.1 --port 8787
```

開啟 `http://127.0.0.1:8787`。OpenAPI contract 位於 `/openapi.json`，metrics 位於 `/metrics`。`8787` 是 EdgeFlow 的專用預設 port，避免與工作區內其他 localhost 服務混淆。

Web App 可一鍵啟停固定版本 llama.cpp／vLLM、建立 workload、screen 候選、提交一個受控 GPU benchmark、取消本機 worker，以及查看 runtime 能力、run validation、raw artifacts、evidence 與 policy。第一次啟動 runtime 可下載 registry 鎖定的模型檔；推論 prompt、結果、SQLite 與 artifacts 仍只留本機。

一次只允許一個 managed runtime 與一個 GPU job。Managed runtime 綁定 `127.0.0.1`，使用程序記憶體中的隨機 API key；控制寫入另需只存在分頁記憶體的 token。server 拒絕非 loopback host、跨來源寫入、超過 1 MiB 的 request，以及任意 shell/path/environment 參數。

公開網站不是控制平面。若後續輸出 GitHub Pages，只能包含經清理且 validated 的唯讀 JSON／圖表，不得啟動本機實驗或讀取私人 artifacts。

## CLI surface

```text
edgeflow inspect [--json]
edgeflow doctor [--strict-optional]
edgeflow workload create ...
edgeflow tune screen --workload ...
edgeflow benchmark run --model-ref ... --workload ... --plan ...
edgeflow experiment plan E05
edgeflow profile --run <run_id> --level torch|nsys|ncu
edgeflow diagnose --profile <profiler_summary.json>
edgeflow validate <artifact_dir>
edgeflow policy build --results <eligible_rows.json> ...
edgeflow policy show <policy.json>
edgeflow kernel validate [--full]
edgeflow serve
```

## Validation gates

| Gate | Enforced invariant |
|---|---|
| G0 Schema | Required artifacts, schemas, IDs, canonical hashes, raw JSONL |
| G1 Environment | GPU scope and no unapproved background GPU process |
| G2 Correctness | Reference parity / no NaN / kernel contract |
| G3 Timing | Warmup split, monotonic timestamps, no profiler contamination |
| G4 Stability | Robust CV and first-vs-last-third drift ≤ 3% |
| G5 Statistics | ≥30 engine requests or ≥100 kernel iterations |
| G6 Quality | Profile-specific hard gate; quantized plans cannot bypass it |
| G7 Provenance | Pinned revision, exact command, git/source state |
| G8 Eligibility | Only measured `PASS` runs may enter policies |

Verdicts are `PASS`, `CONDITIONAL_PASS`, `FAIL`, `INVALID`, and `SKIPPED`. `FAIL`/`INVALID` raw artifacts remain available for audit.

## Repository map

```text
src/edgeflow/
├── api/             localhost FastAPI control/read surface
├── cli/             Typer commands
├── core/            immutable contracts and canonical hashing
├── experiments/     isolated run orchestrator
├── hardware/        RTX/CUDA/software fingerprint + doctor
├── kernels/         correctness-gated Triton optimization
├── local/           typed single-GPU job + allowlisted runtime service managers
├── metrics/         robust statistics and paired bootstrap
├── optimizer/       pruning, objectives, break-even
├── policy/          explainable scoped decision lists
├── profiler/        bounded diagnosis rules
├── runtimes/        PyTorch, compile, llama.cpp, vLLM adapters
├── storage/         SQLite migrations and evidence index
├── validation/      G0–G8 final authority
└── workloads/       exact-token controlled inputs
```

控制平面、資料平面與 presentation plane 的完整依賴設計見 [System Architecture](docs/01_SYSTEM_ARCHITECTURE.md)。

## Evidence and claim rules

- `demo`、`estimated`、profiled latency 不可進 headline。
- correctness 或 quality fail 的 plan 永不排名。
- cold start、compile、capture 與 steady state 分開保存。
- engine-only 與 HTTP end-to-end 不在同一 ranking 比較。
- bottleneck diagnosis 是 `HYPOTHESIS`；matched intervention + mediator 才能升級證據。
- runtime/driver/model fingerprint 改變時 policy 回報 `STALE` 並 fallback。
- RTX 4080 SUPER 的結果不外推成多 GPU 或資料中心 GPU 結論。

## Development

```bash
ruff check src tests
pytest -q
python scripts/validate_package.py
python scripts/verify_results.py
```

Public CI 不需要 GPU，驗證 schemas、tests、lint、secret/model-weight policy。完整 GPU sweep 透過 self-hosted runner 或本機執行。

## Security and data

不提交模型權重、token、`.env`、gated prompt 原文、Nsight binary trace 或 private artifacts。HTTP adapters 只允許 loopback OpenAI-compatible endpoint；正式 run 不使用 `shell=True`，也不在 benchmark 過程安裝套件。詳見 [SECURITY.md](SECURITY.md) 與 [Reproducibility](docs/11_REPRODUCIBILITY_RELEASE_SECURITY.md)。

## License

EdgeFlow 原始碼採 [Apache License 2.0](LICENSE)。模型、資料集與 runtime 保留各自授權；正式 run 必須記錄 revision 與條款。
