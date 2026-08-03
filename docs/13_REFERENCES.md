# 13 · Primary References and Current Technical Sources

> 本清單在計畫建立日 **2026-08-03** 查核。實作正式 repository 時應再次確認版本與支援狀態，並鎖定具體 revision。

## Hardware

- NVIDIA GeForce RTX 4080 SUPER family specifications: 10,240 CUDA cores, 16 GB GDDR6X, Ada Lovelace.
  https://www.nvidia.com/en-my/geforce/graphics-cards/40-series/rtx-4080-family/

## PyTorch

- `torch.compile` modes (`default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`) and options.
  https://docs.pytorch.org/docs/stable/generated/torch.compile.html
- Profiling `torch.compile`, warm-up, graph breaks, compiled regions.
  https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html
- `torch.profiler` CPU/CUDA activities.
  https://docs.pytorch.org/docs/stable/profiler.html
- Dynamo graph breaks and guards.
  https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.dynamo_core_concepts.html
- Ahead-of-time compiled artifact support.
  https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_aot_compile.html

## Triton

- Official Triton tutorials: vector add, fused softmax, matmul, layer norm, attention, group GEMM, persistent matmul.
  https://triton-lang.org/main/getting-started/tutorials/

## llama.cpp

- Main README: CUDA backend, quantization, local inference, hybrid offload.
  https://github.com/ggml-org/llama.cpp/blob/master/README.md
- `llama-bench` fields and output formats.
  https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md
- HTTP server: OpenAI-compatible routes, continuous batching, parallel decoding, monitoring.
  https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

## vLLM

- Current installation/hardware platforms.
  https://docs.vllm.ai/en/stable/getting_started/installation/
- Optimization: chunked prefill and `max_num_batched_tokens` trade-offs.
  https://docs.vllm.ai/en/stable/configuration/optimization/
- Benchmark CLI (`latency`, `serve`, `throughput`).
  https://docs.vllm.ai/en/latest/cli/

## NVIDIA Profilers / WSL

- NVIDIA CUDA on WSL user guide; install the Windows driver and avoid installing a Linux display driver inside WSL.
  https://docs.nvidia.com/cuda/wsl-user-guide/index.html
- PyTorch official local installation selector.
  https://docs.pytorch.org/get-started/locally/
- Nsight Systems CUDA API/workload timeline and CUDA Graph trace.
  https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- Nsight Compute profiling guide.
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- Nsight Compute WSL2 requirements and performance counter permission.
  https://docs.nvidia.com/nsight-compute/2025.3/ReleaseNotes/topics/system-requirements.html
- Microsoft WSL GPU compute setup.
  https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute

## Models

- Qwen3.5 4B official model card.
  https://huggingface.co/Qwen/Qwen3.5-4B
- Llama 3.2 3B Instruct official model card.
  https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- Ministral 3 3B Instruct BF16 official model card.
  https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16
- Ministral 3 3B official GGUF.
  https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF
- Gemma 3 4B IT official model card / QAT GGUF for optional extension.
  https://huggingface.co/google/gemma-3-4b-it
  https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf

## Datasets

- UltraChat 200k.
  https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
- LMSYS-Chat-1M, gated agreement.
  https://huggingface.co/datasets/lmsys/lmsys-chat-1m
- WikiText.
  https://huggingface.co/datasets/Salesforce/wikitext
- ARC.
  https://huggingface.co/datasets/allenai/ai2_arc
- GSM8K.
  https://huggingface.co/datasets/openai/gsm8k
- HumanEval.
  https://huggingface.co/datasets/openai/openai_humaneval
