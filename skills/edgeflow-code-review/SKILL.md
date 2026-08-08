---
name: edgeflow-code-review
description: Review an EdgeFlow change against two independent axes: repository and systems standards, and fidelity to the originating specification or experiment protocol.
version: 1.0.0
---

# EdgeFlow Code Review

## Purpose

Prevent well-written wrong changes and spec-correct poorly designed changes from masking one another. Report the two axes separately.

## Inputs

- fixed comparison point or PR diff;
- originating issue/spec/experiment protocol;
- `AGENTS.md`, `CONTEXT.md`, relevant ADRs;
- repository standards and validation matrix.

## Workflow

### 1. Pin review scope

Record:

- base ref or merge base;
- changed commits and files;
- originating specification;
- relevant validation requirement IDs;
- lane: engineering, formal experiment, or mixed.

Completion: review is against a stable diff and source of intent.

### 2. Standards review

Check:

- canonical domain terms;
- security and local-first boundaries;
- no arbitrary shell/path/environment inputs;
- module depth, locality, and public seam;
- tests observe behavior rather than private implementation;
- no duplicated rule logic or speculative generality;
- raw-artifact immutability and provenance;
- formal timing/profiler separation;
- validation matrix updated when hard behavior changes;
- CI and focused commands.

Classify findings as hard violation or design judgement.

Completion: every material standards finding cites a file/hunk and the relevant rule.

### 3. Specification review

Check:

- missing or partial acceptance criteria;
- behavior outside scope;
- incorrect failure/fallback behavior;
- experiment variables or controls changed;
- artifact or evidence omissions;
- validation status overstated;
- claim wording exceeds evidence level;
- search/holdout leakage;
- changed thresholds without a new protocol.

Completion: every finding cites the requested behavior or protocol requirement.

### 4. Formal-experiment additions

For experiment PRs, also verify:

- preregistration predates confirmatory rows;
- exact model/runtime revisions;
- raw artifact checksums;
- deviations;
- G0–G8 verdicts;
- paired/holdout role;
- quality and correctness;
- allowed and prohibited claims.

### 5. Report separately

Use:

```markdown
## Standards

## Specification

## Verification status
```

Do not collapse findings into one score. End with counts and the most severe finding within each axis.

## Never Do

- Never let passing CI prove specification fidelity.
- Never let a complete experiment spec excuse unsafe or shallow code.
- Never review performance from summary screenshots without raw artifacts.
- Never approve a claim because the wording sounds cautious; inspect evidence level and scope.
- Never silently rerank one review axis over the other.
