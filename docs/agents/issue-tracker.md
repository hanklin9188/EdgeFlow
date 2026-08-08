# Issue Tracker

EdgeFlow uses GitHub Issues in `hanklin9188/EdgeFlow` as the operational work tracker.

## Work-item rules

- One issue delivers one narrow, complete, demonstrable vertical slice.
- Every issue declares `Blocked by` edges.
- Formal experiment issues use `.github/ISSUE_TEMPLATE/formal_experiment.yml`.
- A work item is agent-ready only when its acceptance criteria, public seam, expected artifacts, and validation command are explicit.
- Parent roadmap documents remain planning sources; completed work is tracked in issues and PRs.

## Pull requests

- A PR references its originating issue or specification.
- Engineering PRs require behavior tests and two-axis review.
- Formal experiment PRs require raw-artifact references, validation verdicts, protocol deviations, and claim eligibility.
- PRs remain draft while a blocking validation gate is unresolved.

## Blocking representation

GitHub issue bodies use:

```markdown
## Blocked by

- #123
- #124
```

A ticket with no blockers states:

```markdown
None — ready to start.
```
