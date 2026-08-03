# 11 · Reproducibility, Release, and Security

## 11.1 Reproducibility Unit

EdgeFlow 不把一張圖當作最小可重現單位。最小單位是：

```text
hardware fingerprint
+ model revision/hash
+ dataset revision/sample IDs
+ execution plan
+ workload spec
+ exact command
+ raw per-request metrics
+ validation verdict
```

任何 summary 都必須可以由 raw artifacts 重新生成。

---

## 11.2 Environment Pinning

正式 release 保存：

- `uv.lock`／`requirements.lock`；
- Python version；
- PyTorch wheel build；
- CUDA runtime；
- Dockerfile與image digest；
- llama.cpp commit + CMake flags；
- vLLM version／container digest；
- Nsight versions；
- OS kernel；
- Windows/WSL version。

不能只寫 `latest`。

---

## 11.3 Artifact Layout

```text
runs/
└── 2026-xx-xx/<run_id>/
    ├── manifest.json
    ├── workload.json
    ├── plan.json
    ├── hardware.json
    ├── metrics.jsonl
    ├── summary.json
    ├── validation.json
    ├── stdout.log
    ├── stderr.log
    ├── profiler/
    └── checksums.sha256
```

大型 Nsight trace 不一定全部進git；可放release asset／external artifact storage，repo保留checksum與下載說明。

---

## 11.4 Data Versioning

- Synthetic prompt直接version control。
- Public dataset保存revision與sample IDs。
- Gated dataset只保存不可逆hash與script。
- Processed tables可進git，raw大型檔用release/DVC-like storage。
- 任何手動修正需有transformation log。

---

## 11.5 CI

### Public GitHub-hosted CI

- formatting/lint；
- unit tests；
- JSON/YAML/schema；
- CPU smoke；
- mock runtime adapter；
- no-secret；
- docs links；
- UI build；
- package audit。

### GPU CI

因GitHub-hosted runner未必有合適GPU：

- self-hosted RTX runner optional；
- nightly canary；
- runner label固定；
- secrets與model token隔離；
- fork PR不自動執行self-hosted privileged jobs。

---

## 11.6 Security

### Secrets

禁止commit：

- HF token；
- API key；
- model access cookie；
- `.env`；
- OAuth credential。

使用：

- environment variables；
- OS credential store；
- GitHub Actions secrets；
- `.env.example` only。

### Shell execution

CLI用structured subprocess args，不用 `shell=True` 組接使用者字串。

### Model／dataset trust

- 優先 safetensors；
- `trust_remote_code=False`預設；
- 若必須remote code，pin revision、review、隔離environment並明示；
- dataset arbitrary code loader避免或pin review；
- GGUF hash驗證。

### Generated code

HumanEval執行需：

- container；
- no network；
- CPU/memory/time limit；
- read-only filesystem（必要寫tmp）；
- 不在host直接exec。

### Web service

- localhost強制；CLI拒絕非 loopback bind；
- state-changing control 需要短生命週期、只存於分頁記憶體的 token；
- Host allowlist 與 cross-origin write rejection；
- 無任意file path traversal；
- report API只讀allowlisted artifact root；
- request size limit；
- CSRF/CORS明確；
- benchmark API只接受 registered model + typed schema，不接受 command/environment/output path；
- public deployment不承載 control plane；若未來開放 remote control，需另行設計 auth、TLS、rate limit 與 threat model。

---

## 11.7 Third-Party License

建立：

- `THIRD_PARTY.md`；
- `MODEL_LICENSES.md`；
- `DATA_LICENSES.md`；
- `NOTICE.md`。

EdgeFlow code建議Apache-2.0。模型／資料各自受原條款，不因repo license改變。

---

## 11.8 Release Levels

### v0.1-alpha

- core benchmark；
- single model；
- PyTorch + llama.cpp；
- no headline research claim。

### v0.2-beta

- vLLM；
- causal interventions；
- policy；
- dashboard；
- validated results。

### v1.0

- full confirmatory matrix；
- custom kernel；
- reproduction challenge；
- complete audit；
- stable schemas；
- optional Copilot only if grounded tests pass。

---

## 11.9 Release Checklist

每次release：

1. clean git；
2. lock updated；
3. unit/schema/UI pass；
4. canary GPU pass；
5. model/dataset revisions verified；
6. results regenerated from raw；
7. no demo numbers mislabeled；
8. limitations updated；
9. changelog；
10. tag/sign；
11. release assets checksums；
12. 若有 public result export，驗證其唯讀、清理與 provenance 標示。

---

## 11.10 Result Longevity

所有performance claim都有 expiration condition：

- hardware fingerprint change；
- driver/CUDA major update；
- backend major/minor change with kernel/scheduler impact；
- model revision；
- code change；
- policy older than canary threshold。

UI顯示 `Last validated`，過期標stale。
