---
name: edgeflow-profiler-diagnosis
description: Convert PyTorch, Nsight Systems, Nsight Compute, llama.cpp, and vLLM profiling artifacts into bounded bottleneck hypotheses and controlled intervention plans.
version: 1.0.0
---

# EdgeFlow Profiler Diagnosis Skill

## Principle

Profiler output supports a hypothesis; it does not by itself prove a cause. Every diagnosis must include measurable evidence, confidence, alternative explanations, and a proposed intervention.

## Inputs

- validated or diagnostic run
- normalized phase timing
- kernel summary
- CPU/GPU timeline summary
- memory and clock samples
- backend configuration

## Output

```json
{
  "run_id": "...",
  "observations": [],
  "hypotheses": [],
  "recommended_interventions": [],
  "insufficient_evidence": []
}
```

## Workflow

### 1. Separate Phases

Never diagnose aggregate request only. Split:

- load/compile;
- tokenize/queue;
- prefill;
- first decode;
- steady decode;
- sampling/serialization.

### 2. Inspect Timeline Integrity

Check profiler overhead and NVTX range completeness. If events are missing, say so.

### 3. Generate Observations

Observations must be literal metrics, e.g.:

- `47% of steady decode GPU time is in GEMV/GEMM kernels`;
- `kernel gap ratio is 24%`;
- `peak VRAM reaches 15.2 GiB`;
- `queue delay accounts for 38% of p95 latency`.

### 4. Map to Candidate Bottlenecks

#### Launch

Evidence combination, not one threshold alone:

- short kernels;
- high gaps;
- CPU launch activity;
- low occupancy/SM active;
- batch sensitivity.

#### Memory

- decode dominates;
- high memory throughput or weight traffic proxy;
- low arithmetic intensity;
- quantization/batch evidence.

#### Compute

- long prefill;
- GEMM/attention dominates;
- high tensor/SM utilization;
- gap low.

#### KV Capacity

- VRAM close to limit;
- context/concurrency scaling;
- OOM frontier;
- KV estimate matches growth.

#### Scheduler

- queue/ITL tail;
- mixed prefill/decode;
- token budget sensitivity.

#### Compile

- startup dominated by compile;
- repeated compilation;
- shape guard churn.

### 5. Assign Confidence

Confidence is rule-derived and calibrated; do not output 0.99 casually.

Suggested:

- high: ≥3 independent evidence categories;
- medium: 2;
- low: 1;
- insufficient: contradictory or missing.

### 6. Propose Intervention

Each intervention includes:

- target hypothesis;
- exact changed variable;
- controlled variables;
- expected mediator;
- expected outcome;
- negative control;
- number of repetitions;
- rollback/fallback.

### 7. Do Not Claim Cause Yet

Use wording:

> The trace is consistent with a launch-overhead hypothesis. A CUDA Graph matched intervention should reduce kernel gaps if this explanation is correct.

After intervention, the validation skill decides support/rejection.

## Common Traps

- low GPU utilization does not automatically mean CPU bottleneck;
- high DRAM percentage can be high because compute is low, not because absolute bandwidth is saturated;
- top kernel share does not mean it is optimizable;
- profiler can perturb short kernels;
- quantization changes both memory traffic and kernel implementation;
- batch changes arithmetic intensity and scheduling simultaneously;
- compile mode may change graph, kernels, memory, and capture together; isolate when possible.
