# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for a security problem.

Use GitHub's private vulnerability reporting on this repository
(Security → Report a vulnerability). You should get an initial response within a
week.

## Scope

`proofpath` fetches remote content and can, with explicit consent, download and run a
browser engine. Reports about the following are especially welcome:

- Fetched content escaping its parsing boundary.
- The permission prompt being bypassed, or a large install happening without consent.
- Secrets or the configured contact address leaking into reports, logs or caches.
- Path traversal through a crafted document or archive.

## Out of scope

- A wrong verdict on a claim. That is an accuracy issue — open a normal issue.
- Rate limiting or availability of third-party APIs.
