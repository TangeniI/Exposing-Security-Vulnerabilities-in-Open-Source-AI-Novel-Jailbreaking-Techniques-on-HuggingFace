# Security notes

This is a research/educational repository about AI backdoor attacks. It contains
**no live exploit** and targets only public models with synthetic data.

## Secrets

No credentials are stored in this repository. The HuggingFace token, if used, is
read at runtime from the `HF_TOKEN` environment variable (or Colab secrets) by
`jailbreak_toolkit.config.get_hf_token()` and is never written to disk or printed.

If you fork this project, do not commit tokens. `.gitignore` excludes `.env`,
`*.token`, and `hf_token*` as a safety net.

## Responsible use

The techniques here are intended to help researchers and engineers understand and
**defend against** backdoor / trigger attacks on machine-learning models. Do not
use them against systems you do not own or have explicit permission to test.
