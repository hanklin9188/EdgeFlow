# Dataset Terms and Data Handling

The default smoke and controlled-shape workloads use the version-controlled EdgeFlow synthetic corpus. It is part of this Apache-2.0 project and contains no private prompt material.

Formal real-distribution and quality experiments may use the following upstream datasets:

- `HuggingFaceH4/ultrachat_200k`;
- `Salesforce/wikitext` (`wikitext-2-raw-v1`);
- `allenai/ai2_arc` (`ARC-Challenge`);
- `openai/gsm8k` (`main`);
- optionally gated `lmsys/lmsys-chat-1m`.

Dataset contents are not redistributed by EdgeFlow. Before use, the operator must pin a revision and review the current dataset card, license, agreement, and privacy constraints. Public artifacts contain only permitted sample IDs or irreversible hashes, tokenizer/count metadata, aggregate statistics, and transformation provenance. Gated or private prompt text stays outside the repository.
