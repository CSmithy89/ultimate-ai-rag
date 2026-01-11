# API Gateway Rate Limiting

Purpose: enforce coarse-grained rate limits at the edge in addition to in-app
limits. This reduces abusive traffic before it reaches the service and protects
shared dependencies (Redis, Neo4j, upstream LLMs).

## Recommended Defaults

- Apply a global limit per IP for all routes (e.g., 60 req/min).
- Add tighter per-route limits for high-risk endpoints:
  - `/mcp/ui/config`
  - `/a2a/middleware/*`
  - `/a2a/execute`
- Use burstable limits to avoid punishing short spikes from legitimate clients.

## Nginx Example

```nginx
limit_req_zone $binary_remote_addr zone=api_global:10m rate=60r/m;
limit_req_zone $binary_remote_addr zone=mcp_ui_config:10m rate=30r/m;
limit_req_zone $binary_remote_addr zone=a2a_middleware:10m rate=30r/m;

server {
  location /api/v1/ {
    limit_req zone=api_global burst=20 nodelay;
    proxy_pass http://agentic-rag-backend;
  }

  location /api/v1/mcp/ui/config {
    limit_req zone=mcp_ui_config burst=10 nodelay;
    proxy_pass http://agentic-rag-backend;
  }

  location /api/v1/a2a/middleware/ {
    limit_req zone=a2a_middleware burst=10 nodelay;
    proxy_pass http://agentic-rag-backend;
  }
}
```

## Verification

- Confirm `429` responses are returned under load tests.
- Ensure rate limit logs are visible in gateway logs.
- Validate in-app rate limiting still triggers for tenant-based limits.
