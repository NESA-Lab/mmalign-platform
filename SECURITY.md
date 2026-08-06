# Security and privacy

## Reporting

Please report suspected credential exposure or privacy issues privately to the
maintainers. Do not include secrets, personal data, private prompts, or model
responses in a public issue.

## Repository hygiene

- API credentials must be supplied through environment variables. Never place
  credential values in `configs/models.yaml` or commit a local `.env` file.
- Runtime data under `data/`, including uploaded benchmarks, prompts, model
  responses, review labels, and result exports, is excluded from Git.
- Only synthetic or appropriately licensed benchmark material may be placed
  under `examples/benchmarks/builtin/`.
- Before publishing, inspect staged content with `git diff --cached` and run a
  secret scanner such as Gitleaks or GitHub secret scanning.
- If a secret is committed, revoke and rotate it immediately. Removing it from
  the latest commit does not remove it from Git history.

## Deployment

The development API enables permissive CORS and has no built-in authentication.
Do not expose it directly to the public internet. Production deployments should
add authentication, restrictive CORS, TLS, request-size limits, audit logging,
and retention controls for prompts and model outputs.
