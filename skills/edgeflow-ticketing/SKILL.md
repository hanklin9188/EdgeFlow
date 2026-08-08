---
name: edgeflow-ticketing
description: Convert an approved EdgeFlow specification or experiment program into narrow, complete tracer-bullet issues with explicit blockers, executable acceptance criteria, and evidence outputs.
disable-model-invocation: true
version: 1.0.0
---

# EdgeFlow Ticketing

## Purpose

Turn a large plan into work items that one fresh implementation context can complete and verify. Tickets are vertical slices, not lists of files by layer.

## Inputs

- approved specification, ADR, experiment catalog entry, or backlog section;
- `CONTEXT.md`;
- issue-tracker rules;
- current implementation/evidence state.

## Workflow

### 1. Identify the deliverable graph

List the observable capabilities required for the destination. Separate:

- engineering prerequisites;
- formal experiment prerequisites;
- validation hardening;
- evidence confirmation;
- release/presentation.

Completion: no ticket mixes engineering completion with experimental confirmation unless the slice genuinely delivers both end to end.

### 2. Draft tracer-bullet slices

Each ticket must:

- deliver one demonstrable behavior or evidence artifact;
- cut through schema/API/storage/UI/tests where required;
- fit one fresh context;
- use canonical terms;
- avoid speculative adjacent features.

For formal experiments, one ticket may cover a tightly coupled experiment block when the shared protocol is the actual seam.

Completion: every ticket can be closed independently without leaving an unusable horizontal layer.

### 3. Add blocking edges

A blocker is genuine only when the downstream ticket cannot be validly executed or reviewed before it completes.

Examples:

- timer calibration blocks performance baseline;
- fairness audit blocks cross-runtime ranking;
- formal profiler evidence blocks kernel-target selection;
- holdout data blocks claim confirmation, not exploratory search.

Completion: the ready frontier is unambiguous.

### 4. Write executable acceptance criteria

Engineering criteria name:

- public seam;
- red test;
- focused verification command;
- full CI command;
- failure behavior.

Experiment criteria name:

- protocol ID;
- raw artifacts;
- applicable gates;
- minimum repetitions;
- quality/correctness;
- holdout or mediator requirements;
- exact claim permitted after closure.

Completion: a reviewer can decide done/not-done without interpreting intent.

### 5. Handle wide changes

For one mechanical change with broad blast radius, use:

```text
expand → migrate in green batches → contract
```

Do not disguise a broad breaking rename or schema migration as one vertical ticket.

### 6. Publish

Create issues in blocker-first order. Each issue includes:

- What to build;
- Acceptance criteria;
- Blocked by;
- Evidence or verification outputs;
- Out of scope;
- Suggested lane and labels.

Completion: all no-blocker tickets are immediately actionable.

## Never Do

- Never create a ticket that only says “implement backend” without observable behavior.
- Never create one giant E01–E30 issue.
- Never treat documentation presence as experimental completion.
- Never hide blockers in prose; list them explicitly.
- Never include mutable performance targets before calibration or preregistration.
