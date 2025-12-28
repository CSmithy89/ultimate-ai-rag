# Review Follow-ups

## Implemented
- [x] Fix backend app export so `from agentic_rag_backend import app` works.
- [x] Harden config loading with BACKEND_PORT validation and clearer missing-var errors.
- [x] Prevent frontend volume mount from clobbering node_modules in Docker Compose.
- [x] Make frontend Docker build reproducible with root context + pnpm lockfile.
- [x] Add development-only credential warnings and env defaults.
- [x] Pin core runtime dependencies for reproducible installs.
- [x] Update frontend healthcheck to use Node http instead of fetch.
- [x] Clarify README env setup instructions.
- [x] Add RootLayout return type annotation.
- [x] Add husky pre-commit lint/type-check hooks.
- [x] Add backend lint (ruff) and type-check (mypy) in CI.
- [x] Gemini Code Assist bot is active on the repo (GitHub App).

## Optional / Future
- [ ] Add Dependabot for npm + pip updates.
- [ ] Add CodeQL security scanning.
- [ ] Consider switching backend healthcheck to curl/wget (requires adding it to the image).
- [ ] Add TODOs/notes for deferred service wiring (DB/Redis clients) if needed.
