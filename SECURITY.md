# Security Policy

## Reporting a vulnerability

Please do not report security vulnerabilities or exposed credentials through public GitHub issues.

Instead, report them privately using [GitHub Security Advisories](https://github.com/jayshah5696/finetuning-labs/security/advisories/new) on this repository. Include the affected file paths, commit hashes, and steps to reproduce the issue.

## Credential hygiene

- Never commit API keys, tokens, `.env` files, or private user data to Git or Hugging Face repositories.
- Hosted evaluation and training workflows read secrets from environment variables (`OPENROUTER_API_KEY`) or Google Colab Secrets (`HF_TOKEN`, `WANDB_API_KEY`) at runtime only.
