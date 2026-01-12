# Security Best Practices Guide

This guide provides comprehensive security guidelines and best practices for deploying and operating the Agentic RAG + GraphRAG platform in production environments.

## Table of Contents

- [Authentication and Authorization](#authentication-and-authorization)
  - [API Key Management](#api-key-management)
  - [Multi-Tenancy Enforcement](#multi-tenancy-enforcement)
  - [Rate Limiting Configuration](#rate-limiting-configuration)
- [Data Protection](#data-protection)
  - [Encryption at Rest](#encryption-at-rest)
  - [Encryption in Transit](#encryption-in-transit)
  - [Sensitive Data Handling](#sensitive-data-handling)
- [Network Security](#network-security)
  - [SSRF Protection](#ssrf-protection)
  - [Network Policies](#network-policies)
  - [Firewall Rules](#firewall-rules)
- [Input Validation](#input-validation)
  - [Pydantic Validation](#pydantic-validation)
  - [URL Validation](#url-validation)
  - [File Upload Security](#file-upload-security)
- [Secrets Management](#secrets-management)
  - [Environment Variables](#environment-variables)
  - [API Key Rotation](#api-key-rotation)
  - [Credential Storage](#credential-storage)
- [Compliance](#compliance)
  - [OWASP Considerations](#owasp-considerations)
  - [Data Retention](#data-retention)
  - [Audit Logging](#audit-logging)
- [Security Checklist](#security-checklist)
  - [Pre-Deployment Checklist](#pre-deployment-checklist)
  - [Regular Audit Items](#regular-audit-items)

---

## Authentication and Authorization

### API Key Management

The platform uses API key-based authentication for MCP and A2A protocols with SHA-256 hashing for secure storage.

#### Key Generation

Generate cryptographically secure API keys:

```python
from agentic_rag_backend.mcp_server.auth import generate_api_key

# Generate MCP API key
api_key = generate_api_key(prefix="mcp")  # e.g., mcp_abc123...

# Generate A2A API key
a2a_key = generate_api_key(prefix="a2a")  # e.g., a2a_xyz789...
```

#### Key Storage

API keys are stored as SHA-256 hashes, never in plaintext:

```python
from agentic_rag_backend.mcp_server.auth import MCPAPIKeyAuth

authenticator = MCPAPIKeyAuth()

# Register a key (stores hash, not plaintext)
key_hash = authenticator.register_key(
    api_key="mcp_user_provided_key",
    tenant_id="tenant-uuid",
    scopes=["read", "write"],  # Optional scope restrictions
    is_admin=False,            # Admin keys can access all tenants
)
```

#### Best Practices

| Practice | Description |
|----------|-------------|
| Use unique keys per tenant | Never share API keys across tenants |
| Set appropriate scopes | Limit keys to minimum required permissions |
| Implement key expiration | Rotate keys at regular intervals (90 days recommended) |
| Monitor key usage | Track API key activity for anomaly detection |
| Revoke compromised keys | Have a process to immediately revoke leaked keys |

#### Key Revocation

```python
# Revoke a compromised key immediately
success = authenticator.revoke_key("mcp_compromised_key")
if success:
    logger.info("Key revoked successfully")
```

### Multi-Tenancy Enforcement

Every database query and API operation MUST include `tenant_id` filtering to ensure complete data isolation.

#### Tenant ID Validation

```python
from agentic_rag_backend.validation import is_valid_tenant_id

# Validate tenant_id format (UUID)
tenant_id = "550e8400-e29b-41d4-a716-446655440000"
if not is_valid_tenant_id(tenant_id):
    raise ValueError("Invalid tenant_id format")
```

#### Database Query Patterns

Always include `tenant_id` in queries:

```sql
-- PostgreSQL (correct)
SELECT * FROM trajectories
WHERE tenant_id = %(tenant_id)s AND id = %(trajectory_id)s;

-- Neo4j (correct)
MATCH (e:Episode {tenant_id: $tenant_id})
WHERE e.id = $episode_id
RETURN e;
```

#### Tenant Isolation Testing

The platform includes automated tests for tenant isolation:

```python
# From test suite (story 19-J1)
async def test_cross_tenant_access_blocked():
    """Verify tenant A cannot access tenant B's data."""
    # Attempt cross-tenant access
    response = await client.get(
        f"/api/v1/trajectories/{tenant_b_trajectory_id}",
        headers={"X-Tenant-ID": tenant_a_id}
    )
    assert response.status_code == 404  # Not 403, to avoid enumeration
```

### Rate Limiting Configuration

The platform implements multi-layer rate limiting for protection against abuse.

#### Application-Level Rate Limiting

Configure via environment variables:

```bash
# Global rate limit (requests per minute per tenant)
RATE_LIMIT_PER_MINUTE=60

# Rate limit backend (memory or redis)
RATE_LIMIT_BACKEND=redis

# Redis key prefix for rate limiting
RATE_LIMIT_REDIS_PREFIX=rl

# Retry-After header value (seconds)
RATE_LIMIT_RETRY_AFTER_SECONDS=60
```

#### MCP Server Rate Limiting

```python
from agentic_rag_backend.mcp_server.auth import MCPRateLimiter

# Create rate limiter
rate_limiter = MCPRateLimiter(
    max_requests=60,    # 60 requests
    window_seconds=60,  # per minute
)

# Check if request is allowed
if not await rate_limiter.allow(tenant_id):
    remaining, reset_in = await rate_limiter.get_remaining(tenant_id)
    raise HTTPException(
        status_code=429,
        detail=f"Rate limit exceeded. Retry after {reset_in:.0f}s",
        headers={"Retry-After": str(int(reset_in))}
    )
```

#### A2A Resource Limits

Configure per-tenant session and message limits:

```bash
# A2A Resource Limits
A2A_MAX_SESSIONS_PER_TENANT=100
A2A_MAX_MESSAGES_PER_SESSION=1000
A2A_MESSAGE_RATE_LIMIT=60
A2A_SESSION_TTL_HOURS=24
A2A_LIMITS_BACKEND=redis
```

#### Gateway-Level Rate Limiting

See [API Gateway Rate Limiting Runbook](../runbooks/api-gateway-rate-limiting.md) for nginx configuration.

---

## Data Protection

### Encryption at Rest

#### Trajectory Encryption

Trajectory data (agent thoughts, actions, observations) is encrypted using AES-256-GCM:

```python
from agentic_rag_backend.ops.trace_crypto import TraceCrypto

# Initialize with 32-byte (256-bit) key
crypto = TraceCrypto(key_hex=os.environ["TRACE_ENCRYPTION_KEY"])

# Encrypt sensitive data
encrypted = crypto.encrypt("Agent thought: analyzing user query...")
# Result: "enc:base64_encoded_nonce_and_ciphertext"

# Decrypt for authorized access
plaintext = crypto.decrypt(encrypted)
```

#### Key Generation

Generate a secure encryption key:

```bash
# Generate 32-byte hex key (64 characters)
python -c "import secrets; print(secrets.token_hex(32))"
```

#### Configuration

```bash
# Required for trajectory encryption
TRACE_ENCRYPTION_KEY=64_char_hex_key_here
```

#### Database Encryption

| Database | Encryption Method | Configuration |
|----------|-------------------|---------------|
| PostgreSQL | Transparent Data Encryption (TDE) | Use encrypted storage volumes or PostgreSQL TDE extension |
| Neo4j | Volume-level encryption | Use encrypted EBS/PV or Neo4j Enterprise encryption |
| Redis | Redis encryption at rest | Use Redis Enterprise or encrypted volumes |

### Encryption in Transit

#### TLS Configuration

All external communications must use TLS 1.2 or higher:

```yaml
# Kubernetes Ingress TLS
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
    - hosts:
        - api.yourdomain.com
      secretName: tls-secret
```

#### Internal Service Communication

For Kubernetes deployments, use service mesh mTLS:

```yaml
# Istio PeerAuthentication
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: agentic-rag
spec:
  mtls:
    mode: STRICT
```

#### Database Connection Security

```bash
# PostgreSQL with SSL
DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require

# Neo4j with encryption
NEO4J_URI=neo4j+s://host:7687

# Redis with TLS
REDIS_URL=rediss://host:6379
```

### Sensitive Data Handling

#### Data Classification

| Classification | Examples | Handling |
|---------------|----------|----------|
| Critical | API keys, passwords, encryption keys | Never log, always encrypt |
| Sensitive | Trajectory content, user queries | Encrypt at rest, audit access |
| Internal | Tenant IDs, session IDs | Log for debugging, no external exposure |
| Public | Model names, configuration flags | Safe to log and expose |

#### Logging Guidelines

```python
import structlog

logger = structlog.get_logger(__name__)

# WRONG - Never log sensitive data
logger.info("auth_attempt", api_key=api_key, password=password)

# CORRECT - Log only safe identifiers
logger.info(
    "auth_attempt",
    api_key_hash=api_key[:16] + "...",  # Truncated
    tenant_id=tenant_id,
)
```

#### Frontend Security

See [Frontend Security Checklist](../checklists/frontend-security-checklist.md) for React/Next.js specific guidelines.

Key patterns:

```typescript
// Detect sensitive keys in data
const SENSITIVE_PATTERN = /\b(password|secret|token|key|auth|bearer|jwt|session)\b/i;

// Never include credentials in URLs
// BAD
fetch(`/api/data?token=${apiToken}`);

// GOOD
fetch('/api/data', {
  headers: { Authorization: `Bearer ${apiToken}` },
});
```

---

## Network Security

### SSRF Protection

Server-Side Request Forgery (SSRF) is a critical concern for ingestion endpoints.

#### URL Validation

The platform validates URLs before fetching:

```python
from urllib.parse import urlparse

BLOCKED_HOSTS = {
    "localhost", "127.0.0.1", "0.0.0.0",
    "169.254.169.254",  # AWS metadata
    "metadata.google.internal",  # GCP metadata
}

BLOCKED_SCHEMES = {"file", "ftp", "gopher"}

def validate_url(url: str) -> bool:
    """Validate URL is safe for server-side fetching."""
    try:
        parsed = urlparse(url)

        # Block dangerous schemes
        if parsed.scheme.lower() in BLOCKED_SCHEMES:
            return False

        # Block internal hosts
        if parsed.hostname and parsed.hostname.lower() in BLOCKED_HOSTS:
            return False

        # Block private IP ranges
        import ipaddress
        try:
            ip = ipaddress.ip_address(parsed.hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return False
        except ValueError:
            pass  # Not an IP address, hostname is fine

        return True
    except Exception:
        return False
```

#### Crawler Security

The Crawl4AI integration includes security measures:

```bash
# Enable proxy for external requests
CRAWL4AI_PROXY_URL=http://proxy.internal:8080

# Limit crawl scope
CRAWL4AI_MAX_CONCURRENT=10
CRAWL4AI_RATE_LIMIT=5.0
```

### Network Policies

#### Kubernetes Network Policy

Restrict pod-to-pod communication:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: backend-network-policy
  namespace: agentic-rag
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Only allow traffic from ingress controller
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8000
  egress:
    # Allow DNS
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53
    # Allow databases (internal only)
    - to:
        - podSelector:
            matchLabels:
              app: postgres
      ports:
        - protocol: TCP
          port: 5432
    # Allow external HTTPS (LLM providers)
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - protocol: TCP
          port: 443
```

### Firewall Rules

#### Recommended Firewall Configuration

| Source | Destination | Port | Protocol | Action | Purpose |
|--------|-------------|------|----------|--------|---------|
| Internet | Load Balancer | 443 | TCP | Allow | HTTPS traffic |
| Load Balancer | Backend | 8000 | TCP | Allow | API requests |
| Backend | PostgreSQL | 5432 | TCP | Allow | Database |
| Backend | Neo4j | 7687 | TCP | Allow | Graph database |
| Backend | Redis | 6379 | TCP | Allow | Cache/rate limiting |
| Backend | LLM APIs | 443 | TCP | Allow | AI providers |
| * | * | * | * | Deny | Default deny |

#### Docker Compose Security

For development, limit exposed ports:

```yaml
services:
  postgres:
    ports:
      - "127.0.0.1:5432:5432"  # Only localhost
  neo4j:
    ports:
      - "127.0.0.1:7474:7474"  # Only localhost
      - "127.0.0.1:7687:7687"
```

---

## Input Validation

### Pydantic Validation

All API inputs are validated using Pydantic models:

```python
from pydantic import BaseModel, Field, field_validator
from uuid import UUID

class IngestRequest(BaseModel):
    """Request model for document ingestion."""

    tenant_id: str = Field(..., min_length=36, max_length=36)
    url: str = Field(..., max_length=2048)
    title: str | None = Field(None, max_length=500)

    @field_validator("tenant_id")
    @classmethod
    def validate_tenant_id(cls, v: str) -> str:
        from agentic_rag_backend.validation import is_valid_tenant_id
        if not is_valid_tenant_id(v):
            raise ValueError("Invalid tenant_id format (must be UUID)")
        return v

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must use HTTP or HTTPS scheme")
        return v
```

### URL Validation

Comprehensive URL validation for ingestion:

```python
from urllib.parse import urlparse

def validate_ingestion_url(url: str) -> tuple[bool, str]:
    """Validate URL for document ingestion.

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        parsed = urlparse(url)

        # Check scheme
        if parsed.scheme not in ("http", "https"):
            return False, "Only HTTP/HTTPS URLs are allowed"

        # Check host
        if not parsed.netloc:
            return False, "URL must have a valid host"

        # Block internal addresses (SSRF protection)
        hostname = parsed.hostname.lower() if parsed.hostname else ""
        if hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
            return False, "Internal URLs are not allowed"

        # Check for IP address in private range
        try:
            import ipaddress
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback:
                return False, "Private IP addresses are not allowed"
        except ValueError:
            pass  # Not an IP, OK

        # Length check
        if len(url) > 2048:
            return False, "URL exceeds maximum length (2048 characters)"

        return True, ""

    except Exception as e:
        return False, f"Invalid URL format: {str(e)}"
```

### File Upload Security

#### Configuration

```bash
# Maximum upload size (MB)
MAX_UPLOAD_SIZE_MB=50

# Temporary upload directory
TEMP_UPLOAD_DIR=/tmp/uploads

# Allowed file extensions
ALLOWED_EXTENSIONS=.pdf,.txt,.md,.docx
```

#### Validation

```python
import magic
from pathlib import Path

ALLOWED_MIME_TYPES = {
    "application/pdf": [".pdf"],
    "text/plain": [".txt", ".md"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
}

def validate_upload(file_path: Path, expected_extension: str) -> tuple[bool, str]:
    """Validate uploaded file for security.

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file exists
    if not file_path.exists():
        return False, "File not found"

    # Check file size
    max_size = int(os.environ.get("MAX_UPLOAD_SIZE_MB", 50)) * 1024 * 1024
    if file_path.stat().st_size > max_size:
        return False, f"File exceeds maximum size ({max_size // (1024*1024)}MB)"

    # Validate MIME type matches extension
    mime_type = magic.from_file(str(file_path), mime=True)
    if mime_type not in ALLOWED_MIME_TYPES:
        return False, f"File type not allowed: {mime_type}"

    allowed_extensions = ALLOWED_MIME_TYPES[mime_type]
    if expected_extension.lower() not in allowed_extensions:
        return False, f"Extension mismatch: {expected_extension} not allowed for {mime_type}"

    return True, ""
```

---

## Secrets Management

### Environment Variables

#### Required Secrets

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@host/db` |
| `NEO4J_PASSWORD` | Neo4j password | Strong random password |
| `TRACE_ENCRYPTION_KEY` | 32-byte hex key for trajectory encryption | 64 hex characters |
| `A2A_SIGNING_SECRET` | Secret for A2A message signing | Random string |

#### Never Commit Secrets

```bash
# .gitignore
.env
.env.local
.env.production
*.pem
*.key
secrets/
```

### API Key Rotation

#### Rotation Process

1. **Generate new key**
   ```bash
   python -c "import secrets; print(secrets.token_hex(32))"
   ```

2. **Add new key (dual-key period)**
   - Configure both old and new keys
   - Update dependent services

3. **Verify new key works**
   - Test authentication with new key
   - Monitor for errors

4. **Remove old key**
   - Revoke old key
   - Update secrets management

5. **Audit**
   - Verify old key cannot authenticate
   - Log rotation completion

#### Automated Rotation

For production, use external secrets management:

```yaml
# Kubernetes External Secret (with Vault)
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: agentic-rag-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    kind: ClusterSecretStore
    name: vault-backend
  target:
    name: agentic-rag-secrets
  data:
    - secretKey: OPENAI_API_KEY
      remoteRef:
        key: secret/data/agentic-rag
        property: openai_api_key
```

### Credential Storage

#### Recommended Solutions

| Solution | Use Case | Features |
|----------|----------|----------|
| HashiCorp Vault | Enterprise | Dynamic secrets, rotation, audit |
| AWS Secrets Manager | AWS deployments | AWS integration, rotation |
| Azure Key Vault | Azure deployments | Azure integration, HSM |
| GCP Secret Manager | GCP deployments | GCP integration, versioning |
| Kubernetes Secrets | Simple deployments | Native K8s, base64 encoded |

#### Kubernetes Secrets Best Practices

```yaml
# Use sealed-secrets for GitOps
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: agentic-rag-secrets
spec:
  encryptedData:
    OPENAI_API_KEY: AgB3...encrypted...
```

---

## Compliance

### OWASP Considerations

#### OWASP Top 10 Mitigations

| Risk | Mitigation | Implementation |
|------|------------|----------------|
| A01: Broken Access Control | Multi-tenancy enforcement | `tenant_id` in all queries |
| A02: Cryptographic Failures | AES-256-GCM encryption | `TraceCrypto` for trajectories |
| A03: Injection | Pydantic validation | Input validation on all endpoints |
| A04: Insecure Design | Threat modeling | Security reviews in PRs |
| A05: Security Misconfiguration | Secure defaults | Configuration validation |
| A06: Vulnerable Components | Dependabot | Automated security updates |
| A07: Auth Failures | API key hashing | SHA-256 key storage |
| A08: Data Integrity | Input validation | Pydantic + Zod validation |
| A09: Logging Failures | Structured logging | Audit trail for security events |
| A10: SSRF | URL validation | SSRF protection in crawler |

#### Security Headers

Configure in your reverse proxy or application:

```python
# FastAPI middleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(HTTPSRedirectMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["api.yourdomain.com"])

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response
```

### Data Retention

#### Retention Policies

| Data Type | Retention Period | Deletion Method |
|-----------|------------------|-----------------|
| Trajectories | 90 days (configurable) | Automated cleanup job |
| Session data | 24 hours | TTL-based expiration |
| Audit logs | 1 year | Archive then delete |
| Error logs | 30 days | Log rotation |
| User data | As required by policy | Manual deletion on request |

#### Automated Cleanup

```python
# Trajectory cleanup job
async def cleanup_old_trajectories(
    pool: AsyncConnectionPool,
    retention_days: int = 90,
) -> int:
    """Delete trajectories older than retention period."""
    async with pool.connection() as conn:
        result = await conn.execute(
            """
            DELETE FROM trajectories
            WHERE created_at < NOW() - INTERVAL '%s days'
            """,
            (retention_days,)
        )
        deleted_count = result.rowcount

    logger.info(
        "trajectory_cleanup_completed",
        deleted_count=deleted_count,
        retention_days=retention_days,
    )
    return deleted_count
```

### Audit Logging

#### Security Events to Log

| Event | Log Level | Required Fields |
|-------|-----------|-----------------|
| Authentication success | INFO | tenant_id, api_key_hash (truncated) |
| Authentication failure | WARNING | source_ip, attempted_key_hash |
| Authorization failure | WARNING | tenant_id, resource, action |
| Rate limit exceeded | WARNING | tenant_id, endpoint, limit |
| Session created/closed | INFO | tenant_id, session_id |
| Data access | INFO | tenant_id, resource_type, resource_id |
| Admin action | INFO | admin_key_hash, action, target |

#### Structured Logging Format

```json
{
  "timestamp": "2026-01-13T10:30:00.000000Z",
  "level": "warning",
  "logger": "agentic_rag_backend.mcp_server.auth",
  "event": "mcp_auth_invalid_key",
  "request_id": "req-abc-123",
  "source_ip": "192.168.1.100",
  "key_hash": "abc123def456...",
  "tenant_attempted": "none"
}
```

---

## Security Checklist

### Pre-Deployment Checklist

#### Infrastructure Security

- [ ] **TLS certificates** installed and valid (Let's Encrypt or CA-signed)
- [ ] **HTTPS redirect** enabled for all HTTP traffic
- [ ] **Firewall rules** configured per recommendations
- [ ] **Network policies** deployed (Kubernetes)
- [ ] **Load balancer** configured with rate limiting
- [ ] **DNS** configured with CAA records

#### Application Security

- [ ] **Environment variables** set for all required secrets
- [ ] **API keys** generated with appropriate scopes
- [ ] **Encryption key** (TRACE_ENCRYPTION_KEY) configured (64 hex chars)
- [ ] **Database passwords** are strong and unique
- [ ] **Rate limits** configured appropriately for expected load
- [ ] **Multi-tenancy** validated in test environment

#### Container Security

- [ ] **Non-root user** configured in Dockerfile
- [ ] **Read-only filesystem** enabled where possible
- [ ] **Resource limits** set (CPU, memory)
- [ ] **Security context** configured (no privilege escalation)
- [ ] **Image scanning** enabled in CI/CD pipeline
- [ ] **Base images** are up-to-date

#### Database Security

- [ ] **Connection encryption** (SSL/TLS) enabled
- [ ] **Dedicated users** for each service
- [ ] **Minimum privileges** granted to application users
- [ ] **Backup encryption** enabled
- [ ] **Connection pooling** configured with limits

#### Secrets Management

- [ ] **No secrets in code** or version control
- [ ] **External secrets manager** configured (production)
- [ ] **Rotation schedule** defined for all secrets
- [ ] **Access audit** configured for secrets access

### Regular Audit Items

#### Weekly

- [ ] Review security alerts from Dependabot/CodeQL
- [ ] Check rate limiting effectiveness
- [ ] Review authentication failure logs
- [ ] Verify backup completion

#### Monthly

- [ ] Update dependencies with security patches
- [ ] Review API key usage patterns
- [ ] Audit user/tenant access logs
- [ ] Test disaster recovery procedures
- [ ] Review firewall rules for drift

#### Quarterly

- [ ] Rotate API keys and encryption keys
- [ ] Penetration testing
- [ ] Security training for team
- [ ] Review and update security documentation
- [ ] Compliance audit (if applicable)

#### Annually

- [ ] Full security assessment
- [ ] Third-party security audit
- [ ] Update threat model
- [ ] Review data retention compliance

---

## Incident Response

### Security Incident Procedure

1. **Identify** - Detect and classify the incident
2. **Contain** - Limit the impact (revoke keys, block IPs)
3. **Eradicate** - Remove the threat
4. **Recover** - Restore normal operations
5. **Learn** - Post-incident review and improvements

### Emergency Contacts

Maintain an up-to-date list of:
- Security team contacts
- On-call engineers
- Cloud provider security contacts
- Legal/compliance contacts

### Key Revocation Emergency

```bash
# Immediate key revocation (if compromised)
curl -X POST https://api.yourdomain.com/admin/keys/revoke \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -d '{"key_hash": "compromised_key_hash"}'

# Force session termination for tenant
curl -X POST https://api.yourdomain.com/admin/sessions/terminate \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -d '{"tenant_id": "affected_tenant_id"}'
```

---

## See Also

- [Deployment & Production Guide](./deployment-production.md) - Infrastructure setup
- [Frontend Security Checklist](../checklists/frontend-security-checklist.md) - React/Next.js security
- [API Gateway Rate Limiting](../runbooks/api-gateway-rate-limiting.md) - Edge rate limiting
- [Observability Guide](./observability.md) - Monitoring and alerting
- [Database Administration Guide](./database-administration.md) - Database security

---

## References

- [OWASP Top 10](https://owasp.org/Top10/)
- [CIS Benchmarks](https://www.cisecurity.org/cis-benchmarks)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [Docker Security Best Practices](https://docs.docker.com/develop/security-best-practices/)
- [Kubernetes Security](https://kubernetes.io/docs/concepts/security/)
