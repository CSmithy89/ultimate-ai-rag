# Story 1.3: Docker Compose Development Environment

Status: done

## Story

As a developer,
I want a Docker Compose configuration that orchestrates all services,
so that I can start the entire stack with a single command.

## Acceptance Criteria

1. Given Docker and Docker Compose are installed, when the developer runs `docker compose up -d`, then the following services start successfully: Backend (8000), Frontend (3000), PostgreSQL with pgvector (5432), Neo4j (7474/7687), Redis (6379).
2. Health checks pass for all services.
3. Hot reload is enabled for backend and frontend.

## Tasks / Subtasks

- [x] Add docker-compose.yml with all required services (AC: 1)
  - [x] Configure ports and environment variables
  - [x] Wire up volumes for local development
- [x] Add Dockerfiles for backend and frontend (AC: 1, 3)
- [x] Add health checks for each service (AC: 2)

## Dev Notes

- Backend runs FastAPI via uvicorn with reload enabled.
- Frontend runs Next.js dev server.
- Use pgvector-enabled Postgres image and Neo4j Community image.

### Project Structure Notes

- docker-compose.yml at repo root.
- Dockerfiles in backend/ and frontend/.

### References

- Epic 1 stories and ACs: `_bmad-output/project-planning-artifacts/epics.md#Epic-1`
- Service stack decisions: `_bmad-output/architecture.md#Graph-Database-Neo4j`

## Dev Agent Record

### Agent Model Used

gpt-5

### Debug Log References

### Completion Notes List

- Added docker-compose stack for backend, frontend, Postgres/pgvector, Neo4j, Redis.
- Added Dockerfiles for backend and frontend with dev-friendly defaults.
- Added health checks and documented compose usage in README.

### File List

- docker-compose.yml
- backend/Dockerfile
- frontend/Dockerfile
- README.md

## Senior Developer Review

Outcome: APPROVE

Notes:
- Compose wiring includes all required services, ports, and dev volumes.
- Health checks are reasonable for each service type.
