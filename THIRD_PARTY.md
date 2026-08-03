# Third-Party Components

EdgeFlow source code is Apache-2.0. Third-party runtimes, libraries, models, and datasets retain their own terms. This repository does not vendor model weights, dataset rows, or runtime binaries.

| Component | Use | Version / pin | Upstream license |
|---|---|---|---|
| PyTorch | CUDA inference and reference kernels | Captured per hardware fingerprint | BSD-style |
| Transformers | Model/tokenizer adapters | Captured per hardware fingerprint | Apache-2.0 |
| Triton | Custom GPU kernel | Captured per hardware fingerprint | MIT |
| FastAPI / Starlette / Uvicorn | Local Web API | `uv.lock` | MIT / BSD-3-Clause |
| NumPy | Statistics | `uv.lock` | BSD-3-Clause |
| Pydantic | Typed contracts | `uv.lock` | MIT |
| llama.cpp | Optional isolated GGUF runtime | `b10242`, commit `96278e39fc83e1d97c881e34bcec39ac7ea98820` | MIT |
| vLLM | Optional isolated serving runtime | `v0.26.0`, CUDA 12.9 wheel hash in `specs/runtime_registry.yaml` | Apache-2.0 |

Before a release, regenerate the environment inventory and review upstream license files. A name in this table does not imply that its binary is redistributed by EdgeFlow.
