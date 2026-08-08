---
name: edgeflow-change-design
description: Resolve an ambiguous EdgeFlow engineering or experiment change before implementation; sharpen domain terms, define the public seam, record real trade-offs, and produce an implementation-ready specification.
disable-model-invocation: true
version: 1.0.0
---

# EdgeFlow Change Design

## Purpose

Prevent misaligned implementation by resolving the decision tree before code or formal measurement begins. This skill designs; it does not implement, run a formal experiment, or certify evidence.

## Inputs

- requested change or experiment idea;
- current repository state;
- `CONTEXT.md`;
- relevant ADRs and specifications;
- known constraints and current evidence.

## Workflow

### 1. Classify the lane

Choose one:

- engineering behavior;
- formal experiment;
- validation requirement;
- architecture change;
- public claim or presentation;
- mixed change requiring separate tickets.

Completion: every requested behavior belongs to one lane or is explicitly split.

### 2. Resolve canonical language

Read `CONTEXT.md`. Identify fuzzy, overloaded, or conflicting terms. For every domain concept in the change:

- use an existing canonical term;
- sharpen the definition;
- or add one new term to the glossary.

Do not put implementation details into `CONTEXT.md`.

Completion: issue/spec language and schema/UI naming use the same terms.

### 3. Define observable behavior and the public seam

State:

- who or what calls the behavior;
- the smallest stable interface;
- inputs and outputs;
- success behavior;
- failure and fallback behavior;
- ownership of data and side effects;
- what must remain unchanged.

For an experiment, define the protocol seam: changed variable, controls, mediator, outcome, artifacts, and validation gate.

Completion: a test or formal run can observe the behavior without inspecting private implementation.

### 4. Stress-test concrete scenarios

Cover at least:

- normal case;
- missing capability or artifact;
- invalid input;
- stale fingerprint or version drift;
- cancellation/failure cleanup;
- security boundary;
- sparse or unsupported workload region;
- negative-control case for a causal experiment.

Completion: every scenario has a defined outcome or is consciously out of scope.

### 5. Design twice for load-bearing interfaces

Produce two materially different designs when the seam is hard to reverse. Compare:

- interface size;
- module depth and locality;
- testability;
- failure isolation;
- migration cost;
- evidence and validation impact;
- security and reproducibility.

Completion: one design is selected with rejected alternatives recorded.

### 6. Decide whether an ADR is justified

Create an ADR only when all are true:

- reversal is meaningfully expensive;
- the choice is surprising without context;
- real alternatives and trade-offs existed.

Completion: a justified ADR is added immediately, or the reason for not creating one is clear.

### 7. Produce the specification

The specification must contain:

- problem and motivation;
- canonical terms;
- in-scope and out-of-scope behavior;
- public seam;
- data/artifact flow;
- state/failure transitions;
- security and provenance constraints;
- validation-requirement changes;
- acceptance criteria;
- blockers and rollout;
- exact verification commands.

Completion: `edgeflow-ticketing` can split the specification without inventing missing behavior.

## Output

- updated glossary when needed;
- ADR when justified;
- implementation-ready specification;
- unresolved assumptions clearly marked;
- recommendation for ticket boundaries.

## Never Do

- Never start implementation while the public seam or acceptance behavior is ambiguous.
- Never turn a unit test into evidence for a systems claim.
- Never weaken validation thresholds to make the proposed design pass.
- Never introduce an agent into the benchmark data plane.
- Never create an ADR for a trivial or easily reversible choice.
