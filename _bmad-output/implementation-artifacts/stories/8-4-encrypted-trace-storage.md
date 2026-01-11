# Story 8.4: Encrypted Trace Storage

Status: done

## Story

As a security engineer,
I want reasoning traces to be encrypted at rest,
so that sensitive query content is protected from unauthorized access.

## Acceptance Criteria

1. Given trajectory data is being persisted, when it is written to the database, then sensitive fields are encrypted using AES-256.
2. Given encryption keys are managed securely, when the service starts, then encryption uses a configured key from environment settings.
3. Given encrypted traces are stored, when authorized access is requested, then traces are decrypted before returning.
4. Given multi-tenant data remains isolated, when traces are queried, then tenant_id filtering is enforced.
5. Given encryption is enabled, when tracing occurs, then latency impact is minimal and does not break query execution.

## Tasks / Subtasks

- [ ] Add AES-256 encryption helper (AC: 1, 2)
  - [ ] Add `backend/src/agentic_rag_backend/ops/trace_crypto.py`
  - [ ] Load encryption key from env and validate length

- [ ] Encrypt trajectory events (AC: 1, 5)
  - [ ] Encrypt event content before persistence
  - [ ] Ensure encryption failures do not break request handling

- [ ] Decrypt for authorized ops access (AC: 3, 4)
  - [ ] Decrypt event content in ops trajectory endpoints
  - [ ] Maintain tenant filtering for all trace queries

## Dev Notes

- AES-256-GCM provides authenticated encryption with low overhead.
- Use a prefix marker for encrypted payloads to preserve backward compatibility.
- Keep encryption optional in code paths but always configured in production.

### Project Structure Notes

- Encryption helpers live under `backend/src/agentic_rag_backend/ops/`.
- Trajectory encryption should be implemented in `backend/src/agentic_rag_backend/trajectory.py`.
- Ops endpoints should decrypt only when returning data to authorized viewers.

### References

- Epic 8 Tech Spec: `_bmad-output/epics/epic-8-tech-spec.md`
- Trajectory logger: `backend/src/agentic_rag_backend/trajectory.py`
- Ops routes: `backend/src/agentic_rag_backend/api/routes/ops.py`

## Dev Agent Record

### Agent Model Used
GPT-5 (Codex CLI)

### Debug Log References
None.

### Completion Notes List
1. Added AES-256-GCM trace encryption with prefix marker and safe decrypt fallback.
2. Wired encryption key through settings and lifecycle initialization.
3. Encrypted trajectory event persistence and decrypted on ops reads.

### File List
- `backend/src/agentic_rag_backend/ops/trace_crypto.py`
- `backend/src/agentic_rag_backend/ops/__init__.py`
- `backend/src/agentic_rag_backend/config.py`
- `backend/src/agentic_rag_backend/main.py`
- `backend/src/agentic_rag_backend/trajectory.py`
- `backend/src/agentic_rag_backend/api/routes/ops.py`
- `.env.example`

## Senior Developer Review

Outcome: APPROVE

Notes:
- AES-256-GCM encryption is enforced before persistence and handled with a safe prefix.
- Ops endpoints decrypt only for authorized reads and preserve tenant filtering.
- Encryption failures degrade gracefully without breaking queries.
