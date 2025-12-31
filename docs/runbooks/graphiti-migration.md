# Graphiti Migration Runbook

## Scope
This runbook covers migrating legacy graph data to Graphiti using `backend/scripts/migrate_to_graphiti.py`.

## Pre-Migration Checklist
- Confirm a maintenance window or read-only mode for ingestion.
- Ensure Postgres, Neo4j, and Redis are healthy.
- Verify `OPENAI_API_KEY` and Graphiti models are configured.
- Confirm backup storage path is available and writable.
- Communicate expected behavior to stakeholders (migration is a read-heavy job).

## Execution Steps
1. Create a legacy graph backup for each tenant:
   ```bash
   cd backend
   uv run python scripts/migrate_to_graphiti.py \
     --backup-path ./backups/graphiti \
     --validate
   ```

2. For a single-tenant migration (recommended for first run):
   ```bash
   uv run python scripts/migrate_to_graphiti.py \
     --tenant-id <tenant-uuid> \
     --backup-path ./backups/graphiti \
     --validate
   ```

3. Optional dry run (validates inputs without ingesting):
   ```bash
   uv run python scripts/migrate_to_graphiti.py \
     --tenant-id <tenant-uuid> \
     --dry-run \
     --backup-path ./backups/graphiti \
     --validate
   ```

## Validation
- The script logs entity + relationship counts for legacy vs Graphiti.
- A non-zero exit code indicates validation failure.
- Record counts and store backup locations in the migration report.

## Rollback Procedure
- Restore the legacy graph from the JSONL backup if migration results are invalid.
- Re-enable legacy flags only after confirming rollback completeness.
- Document the rollback reason and corrective actions.

## Troubleshooting
- If validation fails, capture logs and compare entity/relationship counts by tenant.
- If migration is slow, use tenant-scoped runs and confirm batch chunk fetching is enabled.
- If Graphiti errors occur, verify model credentials and Neo4j connectivity.
