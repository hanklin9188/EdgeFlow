# Security Policy

Report vulnerabilities privately through GitHub Security Advisories. Do not open a public issue containing credentials, gated data, private prompts, model weights, or exploit details.

EdgeFlow never requires secrets in repository files. Use environment variables or the Hugging Face credential store; `.env`, tokens, model weights, profiler binaries, private artifacts, and SQLite databases are ignored. Runtime processes use argument arrays without `shell=True`. The API binds to `127.0.0.1` by default and exposes read/planning operations, not arbitrary shell execution.

Model `trust_remote_code` is disabled unless a plan explicitly opts in; such a run requires human review. HumanEval/code execution must run in a network-disabled sandbox with a timeout.
