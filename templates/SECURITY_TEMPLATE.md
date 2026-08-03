# Security Policy

Report vulnerabilities through GitHub private security advisories. Do not open a public issue containing credentials, private prompts, arbitrary-code-execution details, or sensitive profiler artifacts.

## Primary threat surfaces

- command construction for backend workers;
- model and dataset downloads;
- generated-code benchmark execution;
- local HTTP servers;
- artifact upload and path disclosure;
- optional Copilot tool calls.

## Invariants

- no unrestricted shell tool for the Copilot;
- subprocess arguments are structured and allowlisted;
- generated code runs in a network-disabled resource-limited sandbox;
- public artifacts are sanitized;
- model text cannot grant permissions or alter validation policy;
- dashboard is read-only unless an explicit authenticated local control path is enabled.
