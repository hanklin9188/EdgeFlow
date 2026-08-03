# Security Policy

Report vulnerabilities privately through GitHub Security Advisories. Do not open a public issue containing credentials, gated data, private prompts, model weights, or exploit details.

EdgeFlow never requires secrets in repository files. Use environment variables or the Hugging Face credential store; `.env`, tokens, model weights, profiler binaries, private artifacts, and SQLite databases are ignored. Runtime processes use argument arrays without `shell=True`. The API binds to `127.0.0.1` by default and exposes read/planning operations, not arbitrary shell execution.

The Local-first Web App refuses non-loopback bind addresses. Trusted Host, Origin, 1 MiB request-size, in-memory control-token, typed job schema, single-GPU worker, and artifact allowlist checks protect the local control plane. Browser input cannot select an executable, shell command, environment variable, or output path. Managed inference services use fixed launchers, bind to loopback, and receive a random in-memory API key that is inherited only by the isolated worker. Only one managed GPU runtime may be active. A public result export, if created, is a separate read-only surface and never receives either local token.

Model `trust_remote_code` is disabled in Web App submissions. Any future opt-in through audited CLI plans requires human review. HumanEval/code execution must run in a network-disabled sandbox with a timeout.
