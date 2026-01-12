# Deployment & Production Guide

This guide covers deploying the Agentic RAG + GraphRAG platform to production environments, including Docker Compose, Kubernetes, scaling strategies, and security hardening.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Docker Compose Deployment](#docker-compose-deployment)
  - [Development Environment](#development-environment)
  - [Production Environment](#production-environment)
  - [Production Docker Compose Override](#production-docker-compose-override)
- [Kubernetes Deployment](#kubernetes-deployment)
  - [Architecture Overview](#architecture-overview)
  - [Namespace and Resource Setup](#namespace-and-resource-setup)
  - [ConfigMaps and Secrets](#configmaps-and-secrets)
  - [Database Deployments](#database-deployments)
  - [Application Deployments](#application-deployments)
  - [Services and Ingress](#services-and-ingress)
- [Scaling Strategies](#scaling-strategies)
  - [Horizontal Pod Autoscaling](#horizontal-pod-autoscaling)
  - [Backend Scaling](#backend-scaling)
  - [Database Scaling Considerations](#database-scaling-considerations)
- [Security Hardening](#security-hardening)
  - [API Key Management](#api-key-management)
  - [Network Policies](#network-policies)
  - [TLS Configuration](#tls-configuration)
  - [Container Security](#container-security)
- [Monitoring and Observability](#monitoring-and-observability)
  - [Prometheus Integration](#prometheus-integration)
  - [Health Check Endpoints](#health-check-endpoints)
  - [Log Aggregation](#log-aggregation)
- [Load Balancing](#load-balancing)
- [Backup and Disaster Recovery](#backup-and-disaster-recovery)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended (Production) |
|-----------|---------|--------------------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 32+ GB |
| Storage | 50 GB SSD | 200+ GB NVMe SSD |
| Network | 100 Mbps | 1 Gbps |

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Docker | 24.0+ | Container runtime |
| Docker Compose | 2.20+ | Multi-container orchestration |
| Kubernetes | 1.28+ | Container orchestration (optional) |
| kubectl | 1.28+ | Kubernetes CLI |
| Helm | 3.12+ | Kubernetes package manager (optional) |

### External Services

| Service | Required | Notes |
|---------|----------|-------|
| LLM Provider | Yes | OpenAI, Anthropic, OpenRouter, Ollama, or Gemini |
| Embedding Provider | Yes | Same as LLM or Voyage AI |
| Domain + SSL | Recommended | For TLS termination |
| Object Storage | Optional | For backups (S3, GCS, Azure Blob) |

---

## Docker Compose Deployment

### Development Environment

The default `docker-compose.yml` is configured for development with hot-reloading and debug settings.

```bash
# Clone repository
git clone https://github.com/your-org/agentic-rag-graphrag.git
cd agentic-rag-graphrag

# Copy environment template
cp .env.example .env

# Edit configuration
# - Set API keys (OPENAI_API_KEY, etc.)
# - Configure database passwords

# Start development stack
docker compose up -d

# View logs
docker compose logs -f

# Check service health
docker compose ps
```

**Development Services:**

| Service | Port | URL |
|---------|------|-----|
| Frontend | 3000 | http://localhost:3000 |
| Backend | 8000 | http://localhost:8000 |
| PostgreSQL | 5432 | postgres://localhost:5432 |
| Neo4j Browser | 7474 | http://localhost:7474 |
| Neo4j Bolt | 7687 | bolt://localhost:7687 |
| Redis | 6379 | redis://localhost:6379 |

### Production Environment

For production deployments, create a `docker-compose.prod.yml` override file.

**Key Production Changes:**

1. Remove hot-reloading (`--reload` flag)
2. Use production-optimized images
3. Enable resource limits
4. Configure proper logging drivers
5. Use external volumes for persistence
6. Enable health checks with stricter intervals

### Production Docker Compose Override

Create `docker-compose.prod.yml`:

```yaml
# docker-compose.prod.yml
# Usage: docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

services:
  backend:
    # Override development command
    command: uvicorn agentic_rag_backend.main:app --host 0.0.0.0 --port 8000 --workers 4
    # Remove volume mounts (no hot reload)
    volumes: []
    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
    # Production logging
    logging:
      driver: json-file
      options:
        max-size: "100m"
        max-file: "5"
    # Stricter health check
    healthcheck:
      test: ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health')\""]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    environment:
      - APP_ENV=production
      - LOG_LEVEL=info
    restart: always

  frontend:
    # Build production image
    build:
      context: .
      dockerfile: frontend/Dockerfile.prod
    command: pnpm start
    volumes: []
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    logging:
      driver: json-file
      options:
        max-size: "50m"
        max-file: "5"
    healthcheck:
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    restart: always

  postgres:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
    logging:
      driver: json-file
      options:
        max-size: "100m"
        max-file: "10"
    # Use external volume for production
    volumes:
      - postgres-prod-data:/var/lib/postgresql/data
    command:
      - "postgres"
      - "-c"
      - "shared_buffers=2GB"
      - "-c"
      - "effective_cache_size=6GB"
      - "-c"
      - "maintenance_work_mem=512MB"
      - "-c"
      - "checkpoint_completion_target=0.9"
      - "-c"
      - "wal_buffers=64MB"
      - "-c"
      - "default_statistics_target=100"
      - "-c"
      - "random_page_cost=1.1"
      - "-c"
      - "effective_io_concurrency=200"
      - "-c"
      - "work_mem=16MB"
      - "-c"
      - "huge_pages=try"
      - "-c"
      - "max_connections=200"
    restart: always

  neo4j:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
    volumes:
      - neo4j-prod-data:/data
      - neo4j-prod-logs:/logs
    environment:
      - NEO4J_AUTH=${NEO4J_USER}/${NEO4J_PASSWORD}
      - NEO4J_dbms_memory_heap_initial__size=2G
      - NEO4J_dbms_memory_heap_max__size=4G
      - NEO4J_dbms_memory_pagecache_size=2G
    logging:
      driver: json-file
      options:
        max-size: "100m"
        max-file: "10"
    restart: always

  redis:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
    volumes:
      - redis-prod-data:/data
    command: redis-server --appendonly yes --maxmemory 1gb --maxmemory-policy allkeys-lru
    logging:
      driver: json-file
      options:
        max-size: "50m"
        max-file: "5"
    restart: always

volumes:
  postgres-prod-data:
    external: true
  neo4j-prod-data:
    external: true
  neo4j-prod-logs:
    external: true
  redis-prod-data:
    external: true
```

**Production Deployment Commands:**

```bash
# Create external volumes (once)
docker volume create postgres-prod-data
docker volume create neo4j-prod-data
docker volume create neo4j-prod-logs
docker volume create redis-prod-data

# Deploy production stack
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale backend workers
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale backend=3

# Rolling update
docker compose -f docker-compose.yml -f docker-compose.prod.yml pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --no-deps backend frontend
```

**Production Frontend Dockerfile (`frontend/Dockerfile.prod`):**

```dockerfile
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json pnpm-workspace.yaml pnpm-lock.yaml ./
COPY frontend/package.json ./frontend/package.json

RUN corepack enable \
  && corepack prepare pnpm@9.1.0 --activate \
  && pnpm install --filter agentic-rag-frontend... --frozen-lockfile

COPY frontend ./frontend

WORKDIR /app/frontend
RUN pnpm build

# Production image
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/frontend/.next/standalone ./
COPY --from=builder /app/frontend/.next/static ./frontend/.next/static
COPY --from=builder /app/frontend/public ./frontend/public

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "frontend/server.js"]
```

---

## Kubernetes Deployment

### Architecture Overview

```
                    ┌─────────────────┐
                    │    Ingress      │
                    │   (TLS + LB)    │
                    └────────┬────────┘
                             │
            ┌────────────────┴────────────────┐
            │                                 │
     ┌──────▼──────┐                  ┌───────▼──────┐
     │  Frontend   │                  │   Backend    │
     │  Service    │                  │   Service    │
     │ (ClusterIP) │                  │ (ClusterIP)  │
     └──────┬──────┘                  └───────┬──────┘
            │                                 │
     ┌──────▼──────┐                  ┌───────▼──────┐
     │  Frontend   │                  │   Backend    │
     │ Deployment  │◄────────────────►│ Deployment   │
     │ (replicas)  │                  │ (replicas)   │
     └─────────────┘                  └───────┬──────┘
                                              │
            ┌──────────────┬──────────────────┼──────────────┐
            │              │                  │              │
     ┌──────▼──────┐┌──────▼──────┐   ┌───────▼──────┐┌──────▼──────┐
     │ PostgreSQL  ││   Neo4j     │   │    Redis     ││  Prometheus │
     │ StatefulSet ││ StatefulSet │   │ StatefulSet  ││ Deployment  │
     └─────────────┘└─────────────┘   └──────────────┘└─────────────┘
```

### Namespace and Resource Setup

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: agentic-rag
  labels:
    app.kubernetes.io/name: agentic-rag
    app.kubernetes.io/component: platform
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: agentic-rag-quota
  namespace: agentic-rag
spec:
  hard:
    requests.cpu: "20"
    requests.memory: 64Gi
    limits.cpu: "40"
    limits.memory: 128Gi
    persistentvolumeclaims: "10"
---
apiVersion: v1
kind: LimitRange
metadata:
  name: agentic-rag-limits
  namespace: agentic-rag
spec:
  limits:
    - default:
        cpu: "1"
        memory: 2Gi
      defaultRequest:
        cpu: "250m"
        memory: 512Mi
      type: Container
```

### ConfigMaps and Secrets

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: agentic-rag-config
  namespace: agentic-rag
data:
  APP_ENV: "production"
  LOG_LEVEL: "info"
  BACKEND_HOST: "0.0.0.0"
  BACKEND_PORT: "8000"
  FRONTEND_URL: "https://app.yourdomain.com"
  # Database connection strings (without credentials)
  DATABASE_HOST: "postgres-service"
  DATABASE_PORT: "5432"
  DATABASE_NAME: "agentic_rag"
  NEO4J_HOST: "neo4j-service"
  NEO4J_BOLT_PORT: "7687"
  REDIS_HOST: "redis-service"
  REDIS_PORT: "6379"
  # Feature flags
  PROMETHEUS_ENABLED: "true"
  PROMETHEUS_PATH: "/metrics"
  METRICS_TENANT_LABEL_MODE: "hash"
  METRICS_TENANT_LABEL_BUCKETS: "100"
  # Pool settings
  DB_POOL_MIN: "5"
  DB_POOL_MAX: "25"
  NEO4J_POOL_MIN: "2"
  NEO4J_POOL_MAX: "50"
---
# k8s/secrets.yaml (use sealed-secrets or external-secrets in production)
apiVersion: v1
kind: Secret
metadata:
  name: agentic-rag-secrets
  namespace: agentic-rag
type: Opaque
stringData:
  # IMPORTANT: Use external secret management in production!
  # Options: HashiCorp Vault, AWS Secrets Manager, Azure Key Vault,
  #          GCP Secret Manager, Kubernetes External Secrets
  OPENAI_API_KEY: "sk-..."
  POSTGRES_PASSWORD: "secure-password-here"
  NEO4J_PASSWORD: "secure-password-here"
  TRACE_ENCRYPTION_KEY: "64-char-hex-key-here"
  A2A_SIGNING_SECRET: "signing-secret-here"
```

### Database Deployments

```yaml
# k8s/postgres.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: agentic-rag
spec:
  serviceName: postgres-service
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
        - name: postgres
          image: pgvector/pgvector:pg16
          ports:
            - containerPort: 5432
          env:
            - name: POSTGRES_DB
              valueFrom:
                configMapKeyRef:
                  name: agentic-rag-config
                  key: DATABASE_NAME
            - name: POSTGRES_USER
              value: "agentic_rag"
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: agentic-rag-secrets
                  key: POSTGRES_PASSWORD
          args:
            - "-c"
            - "shared_buffers=2GB"
            - "-c"
            - "effective_cache_size=6GB"
            - "-c"
            - "max_connections=200"
          resources:
            requests:
              cpu: "2"
              memory: 4Gi
            limits:
              cpu: "4"
              memory: 8Gi
          volumeMounts:
            - name: postgres-data
              mountPath: /var/lib/postgresql/data
          livenessProbe:
            exec:
              command: ["pg_isready", "-U", "agentic_rag"]
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            exec:
              command: ["pg_isready", "-U", "agentic_rag"]
            initialDelaySeconds: 5
            periodSeconds: 5
  volumeClaimTemplates:
    - metadata:
        name: postgres-data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: fast-ssd
        resources:
          requests:
            storage: 100Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: agentic-rag
spec:
  selector:
    app: postgres
  ports:
    - port: 5432
      targetPort: 5432
  clusterIP: None
```

```yaml
# k8s/neo4j.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: neo4j
  namespace: agentic-rag
spec:
  serviceName: neo4j-service
  replicas: 1
  selector:
    matchLabels:
      app: neo4j
  template:
    metadata:
      labels:
        app: neo4j
    spec:
      containers:
        - name: neo4j
          image: neo4j:5-community
          ports:
            - containerPort: 7474
              name: http
            - containerPort: 7687
              name: bolt
          env:
            - name: NEO4J_AUTH
              valueFrom:
                secretKeyRef:
                  name: agentic-rag-secrets
                  key: NEO4J_PASSWORD
            - name: NEO4J_dbms_memory_heap_initial__size
              value: "2G"
            - name: NEO4J_dbms_memory_heap_max__size
              value: "4G"
            - name: NEO4J_dbms_memory_pagecache_size
              value: "2G"
          resources:
            requests:
              cpu: "2"
              memory: 4Gi
            limits:
              cpu: "4"
              memory: 8Gi
          volumeMounts:
            - name: neo4j-data
              mountPath: /data
          livenessProbe:
            httpGet:
              path: /
              port: 7474
            initialDelaySeconds: 60
            periodSeconds: 30
          readinessProbe:
            tcpSocket:
              port: 7687
            initialDelaySeconds: 30
            periodSeconds: 10
  volumeClaimTemplates:
    - metadata:
        name: neo4j-data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: fast-ssd
        resources:
          requests:
            storage: 50Gi
---
apiVersion: v1
kind: Service
metadata:
  name: neo4j-service
  namespace: agentic-rag
spec:
  selector:
    app: neo4j
  ports:
    - port: 7474
      targetPort: 7474
      name: http
    - port: 7687
      targetPort: 7687
      name: bolt
  clusterIP: None
```

```yaml
# k8s/redis.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
  namespace: agentic-rag
spec:
  serviceName: redis-service
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
        - name: redis
          image: redis:7-alpine
          ports:
            - containerPort: 6379
          command:
            - redis-server
            - --appendonly
            - "yes"
            - --maxmemory
            - "1gb"
            - --maxmemory-policy
            - "allkeys-lru"
          resources:
            requests:
              cpu: "500m"
              memory: 1Gi
            limits:
              cpu: "2"
              memory: 2Gi
          volumeMounts:
            - name: redis-data
              mountPath: /data
          livenessProbe:
            exec:
              command: ["redis-cli", "ping"]
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            exec:
              command: ["redis-cli", "ping"]
            initialDelaySeconds: 5
            periodSeconds: 5
  volumeClaimTemplates:
    - metadata:
        name: redis-data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: fast-ssd
        resources:
          requests:
            storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: agentic-rag
spec:
  selector:
    app: redis
  ports:
    - port: 6379
      targetPort: 6379
  clusterIP: None
```

### Application Deployments

```yaml
# k8s/backend.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
  namespace: agentic-rag
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: agentic-rag-backend
      containers:
        - name: backend
          image: your-registry/agentic-rag-backend:latest
          ports:
            - containerPort: 8000
          env:
            - name: APP_ENV
              valueFrom:
                configMapKeyRef:
                  name: agentic-rag-config
                  key: APP_ENV
            - name: DATABASE_URL
              value: "postgresql://agentic_rag:$(POSTGRES_PASSWORD)@postgres-service:5432/agentic_rag"
            - name: NEO4J_URI
              value: "bolt://neo4j-service:7687"
            - name: NEO4J_USER
              value: "neo4j"
            - name: NEO4J_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: agentic-rag-secrets
                  key: NEO4J_PASSWORD
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: agentic-rag-secrets
                  key: POSTGRES_PASSWORD
            - name: REDIS_URL
              value: "redis://redis-service:6379"
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: agentic-rag-secrets
                  key: OPENAI_API_KEY
            - name: TRACE_ENCRYPTION_KEY
              valueFrom:
                secretKeyRef:
                  name: agentic-rag-secrets
                  key: TRACE_ENCRYPTION_KEY
          envFrom:
            - configMapRef:
                name: agentic-rag-config
          command:
            - uvicorn
            - agentic_rag_backend.main:app
            - --host
            - "0.0.0.0"
            - --port
            - "8000"
            - --workers
            - "4"
          resources:
            requests:
              cpu: "1"
              memory: 2Gi
            limits:
              cpu: "4"
              memory: 8Gi
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 30
            timeoutSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
            timeoutSeconds: 5
          securityContext:
            runAsNonRoot: true
            runAsUser: 1000
            readOnlyRootFilesystem: true
            allowPrivilegeEscalation: false
---
apiVersion: v1
kind: Service
metadata:
  name: backend-service
  namespace: agentic-rag
spec:
  selector:
    app: backend
  ports:
    - port: 8000
      targetPort: 8000
  type: ClusterIP
```

```yaml
# k8s/frontend.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  namespace: agentic-rag
spec:
  replicas: 2
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
        - name: frontend
          image: your-registry/agentic-rag-frontend:latest
          ports:
            - containerPort: 3000
          env:
            - name: NODE_ENV
              value: "production"
            - name: NEXT_PUBLIC_API_URL
              value: "https://api.yourdomain.com"
          resources:
            requests:
              cpu: "500m"
              memory: 1Gi
            limits:
              cpu: "2"
              memory: 4Gi
          livenessProbe:
            httpGet:
              path: /
              port: 3000
            initialDelaySeconds: 30
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /
              port: 3000
            initialDelaySeconds: 10
            periodSeconds: 10
          securityContext:
            runAsNonRoot: true
            runAsUser: 1001
            readOnlyRootFilesystem: true
            allowPrivilegeEscalation: false
---
apiVersion: v1
kind: Service
metadata:
  name: frontend-service
  namespace: agentic-rag
spec:
  selector:
    app: frontend
  ports:
    - port: 3000
      targetPort: 3000
  type: ClusterIP
```

### Services and Ingress

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: agentic-rag-ingress
  namespace: agentic-rag
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts:
        - app.yourdomain.com
        - api.yourdomain.com
      secretName: agentic-rag-tls
  rules:
    - host: app.yourdomain.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: frontend-service
                port:
                  number: 3000
    - host: api.yourdomain.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: backend-service
                port:
                  number: 8000
```

---

## Scaling Strategies

### Horizontal Pod Autoscaling

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
  namespace: agentic-rag
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 25
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 100
          periodSeconds: 30
        - type: Pods
          value: 4
          periodSeconds: 30
      selectPolicy: Max
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: frontend-hpa
  namespace: agentic-rag
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: frontend
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

### Backend Scaling

**Scaling Considerations:**

| Factor | Recommendation |
|--------|----------------|
| CPU-bound (embeddings) | Increase `--workers` flag or pod replicas |
| Memory-bound (large documents) | Increase memory limits |
| I/O-bound (database) | Optimize connection pools |
| LLM latency | Use async processing, increase replicas |

**Uvicorn Worker Scaling:**

```bash
# Formula: workers = (2 * CPU cores) + 1
# For a 4-core pod:
uvicorn agentic_rag_backend.main:app --workers 9
```

**Connection Pool Tuning:**

```bash
# Per-replica pool settings (prevent exhaustion)
# With 10 replicas, each with DB_POOL_MAX=25 = 250 total connections
DB_POOL_MIN=2
DB_POOL_MAX=25

# Neo4j (high-cardinality graph queries)
NEO4J_POOL_MIN=2
NEO4J_POOL_MAX=50
```

### Database Scaling Considerations

#### PostgreSQL Scaling

**Read Replicas (Horizontal):**

For read-heavy workloads, consider PostgreSQL streaming replication:

```yaml
# Example: Using Zalando Postgres Operator
apiVersion: acid.zalan.do/v1
kind: postgresql
metadata:
  name: postgres-cluster
  namespace: agentic-rag
spec:
  teamId: "agentic-rag"
  numberOfInstances: 3
  volume:
    size: 100Gi
    storageClass: fast-ssd
  postgresql:
    version: "16"
    parameters:
      shared_buffers: "2GB"
      effective_cache_size: "6GB"
```

**Vertical Scaling:**

| Workload | CPU | Memory | Storage |
|----------|-----|--------|---------|
| Small (< 100k chunks) | 2 | 4 GB | 50 GB |
| Medium (100k-1M chunks) | 4 | 8 GB | 200 GB |
| Large (1M+ chunks) | 8 | 16 GB | 500 GB+ |

#### Neo4j Scaling

**Causal Cluster (Enterprise):**

For high-availability and read scaling, use Neo4j Causal Clustering:

```yaml
# Requires Neo4j Enterprise license
NEO4J_causal__clustering_minimum__core__cluster__size__at__formation: 3
NEO4J_causal__clustering_minimum__core__cluster__size__at__runtime: 3
NEO4J_causal__clustering_initial__discovery__members: "neo4j-0:5000,neo4j-1:5000,neo4j-2:5000"
```

**Vertical Scaling (Community):**

| Graph Size | CPU | Heap | Page Cache |
|------------|-----|------|------------|
| < 1M nodes | 2 | 2 GB | 2 GB |
| 1M-10M nodes | 4 | 4 GB | 4 GB |
| 10M+ nodes | 8+ | 8 GB | 8 GB+ |

#### Redis Scaling

**Redis Cluster (High Availability):**

```yaml
# Using Redis Operator
apiVersion: redis.redis.opstreelabs.in/v1beta1
kind: RedisCluster
metadata:
  name: redis-cluster
  namespace: agentic-rag
spec:
  clusterSize: 3
  clusterVersion: v7
  persistenceEnabled: true
  storage:
    volumeClaimTemplate:
      spec:
        storageClassName: fast-ssd
        resources:
          requests:
            storage: 10Gi
```

---

## Security Hardening

### API Key Management

**Best Practices:**

1. **Never commit secrets to version control**
2. **Use external secret management:**
   - HashiCorp Vault
   - AWS Secrets Manager
   - Azure Key Vault
   - GCP Secret Manager
   - Kubernetes External Secrets Operator

**External Secrets Example:**

```yaml
# k8s/external-secret.yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: agentic-rag-secrets
  namespace: agentic-rag
spec:
  refreshInterval: 1h
  secretStoreRef:
    kind: ClusterSecretStore
    name: vault-backend
  target:
    name: agentic-rag-secrets
    creationPolicy: Owner
  data:
    - secretKey: OPENAI_API_KEY
      remoteRef:
        key: secret/data/agentic-rag
        property: openai_api_key
    - secretKey: POSTGRES_PASSWORD
      remoteRef:
        key: secret/data/agentic-rag
        property: postgres_password
    - secretKey: NEO4J_PASSWORD
      remoteRef:
        key: secret/data/agentic-rag
        property: neo4j_password
    - secretKey: TRACE_ENCRYPTION_KEY
      remoteRef:
        key: secret/data/agentic-rag
        property: trace_encryption_key
```

**Key Rotation:**

```bash
# Generate new encryption key
python -c "import secrets; print(secrets.token_hex(32))"

# Update secret (via secret manager or kubectl)
kubectl create secret generic agentic-rag-secrets \
  --from-literal=TRACE_ENCRYPTION_KEY=<new-key> \
  --dry-run=client -o yaml | kubectl apply -f -

# Rolling restart to pick up new secrets
kubectl rollout restart deployment/backend -n agentic-rag
```

### Network Policies

```yaml
# k8s/network-policy.yaml
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
    # Allow traffic from ingress controller
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8000
    # Allow traffic from frontend
    - from:
        - podSelector:
            matchLabels:
              app: frontend
      ports:
        - protocol: TCP
          port: 8000
    # Allow Prometheus scraping
    - from:
        - namespaceSelector:
            matchLabels:
              name: monitoring
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
    # Allow PostgreSQL
    - to:
        - podSelector:
            matchLabels:
              app: postgres
      ports:
        - protocol: TCP
          port: 5432
    # Allow Neo4j
    - to:
        - podSelector:
            matchLabels:
              app: neo4j
      ports:
        - protocol: TCP
          port: 7687
    # Allow Redis
    - to:
        - podSelector:
            matchLabels:
              app: redis
      ports:
        - protocol: TCP
          port: 6379
    # Allow external HTTPS (LLM providers)
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - protocol: TCP
          port: 443
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: database-network-policy
  namespace: agentic-rag
spec:
  podSelector:
    matchLabels:
      app: postgres
  policyTypes:
    - Ingress
  ingress:
    # Only allow backend pods
    - from:
        - podSelector:
            matchLabels:
              app: backend
      ports:
        - protocol: TCP
          port: 5432
```

### TLS Configuration

**Cert-Manager with Let's Encrypt:**

```yaml
# k8s/cert-manager.yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@yourdomain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
      - http01:
          ingress:
            class: nginx
---
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: agentic-rag-tls
  namespace: agentic-rag
spec:
  secretName: agentic-rag-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
    - app.yourdomain.com
    - api.yourdomain.com
```

**Internal TLS (mTLS with Istio/Linkerd):**

```yaml
# For service mesh mTLS
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: agentic-rag
spec:
  mtls:
    mode: STRICT
```

### Container Security

**Pod Security Standards:**

```yaml
# k8s/pod-security.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: agentic-rag
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

**Security Context (per container):**

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  fsGroup: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
  seccompProfile:
    type: RuntimeDefault
```

---

## Monitoring and Observability

### Prometheus Integration

**ServiceMonitor for Prometheus Operator:**

```yaml
# k8s/servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: backend-monitor
  namespace: agentic-rag
  labels:
    release: prometheus
spec:
  selector:
    matchLabels:
      app: backend
  endpoints:
    - port: http
      path: /metrics
      interval: 30s
      scrapeTimeout: 10s
```

**Key Metrics to Monitor:**

| Metric | Alert Threshold | Description |
|--------|-----------------|-------------|
| `retrieval_latency_seconds` | p95 > 2s | Retrieval performance |
| `agui_stream_completed_total{status="error"}` | > 5% | Stream errors |
| `llm_api_cost_total` | > $10/hour | Cost control |
| `active_retrieval_operations` | > 50 | Concurrent load |

See [Observability Guide](./observability.md) for complete metrics documentation.

### Health Check Endpoints

| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `GET /health` | Liveness probe | `{"status": "ok"}` |
| `GET /ready` | Readiness probe | `{"status": "ready", "databases": {...}}` |
| `GET /metrics` | Prometheus metrics | Prometheus text format |

**Kubernetes Probe Configuration:**

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 30
  timeoutSeconds: 10
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

### Log Aggregation

**Structured Logging (JSON):**

```json
{
  "timestamp": "2026-01-12T10:30:00.000000Z",
  "level": "info",
  "logger": "agentic_rag_backend.api",
  "event": "request_completed",
  "request_id": "req-abc-123",
  "tenant_id": "acme-corp",
  "duration_ms": 245,
  "status_code": 200
}
```

**Fluentd/Fluent Bit Configuration:**

```yaml
# k8s/fluent-bit.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-config
  namespace: logging
data:
  fluent-bit.conf: |
    [SERVICE]
        Flush         5
        Log_Level     info
        Parsers_File  parsers.conf

    [INPUT]
        Name              tail
        Path              /var/log/containers/backend-*.log
        Parser            docker
        Tag               agentic-rag.backend.*

    [FILTER]
        Name              kubernetes
        Match             agentic-rag.*
        Kube_URL          https://kubernetes.default.svc:443
        Kube_CA_File      /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        Kube_Token_File   /var/run/secrets/kubernetes.io/serviceaccount/token

    [OUTPUT]
        Name              es
        Match             agentic-rag.*
        Host              elasticsearch-master
        Port              9200
        Index             agentic-rag-logs
        Type              _doc
```

---

## Load Balancing

### Layer 7 Load Balancing (Ingress)

**NGINX Ingress Configuration:**

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: agentic-rag-ingress
  namespace: agentic-rag
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-buffering: "off"
    nginx.ingress.kubernetes.io/upstream-hash-by: "$remote_addr"
    # Rate limiting
    nginx.ingress.kubernetes.io/limit-rps: "100"
    nginx.ingress.kubernetes.io/limit-connections: "50"
    # CORS
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "https://app.yourdomain.com"
```

### Layer 4 Load Balancing (Service)

For internal services, Kubernetes Services provide load balancing:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: backend-service
  namespace: agentic-rag
spec:
  selector:
    app: backend
  ports:
    - port: 8000
      targetPort: 8000
  type: ClusterIP
  # Session affinity for stateful operations
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 3600
```

---

## Backup and Disaster Recovery

### Automated Backups

**PostgreSQL Backup CronJob:**

```yaml
# k8s/backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
  namespace: agentic-rag
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: backup
              image: postgres:16
              command:
                - /bin/sh
                - -c
                - |
                  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
                  pg_dump -h postgres-service -U agentic_rag -d agentic_rag -F c \
                    -f /backup/postgres_${TIMESTAMP}.dump
                  # Upload to S3 (using aws cli)
                  aws s3 cp /backup/postgres_${TIMESTAMP}.dump \
                    s3://your-backup-bucket/postgres/postgres_${TIMESTAMP}.dump
                  # Keep only last 7 days locally
                  find /backup -name "*.dump" -mtime +7 -delete
              env:
                - name: PGPASSWORD
                  valueFrom:
                    secretKeyRef:
                      name: agentic-rag-secrets
                      key: POSTGRES_PASSWORD
              volumeMounts:
                - name: backup-volume
                  mountPath: /backup
          volumes:
            - name: backup-volume
              persistentVolumeClaim:
                claimName: backup-pvc
          restartPolicy: OnFailure
```

### Disaster Recovery Plan

| Component | RPO | RTO | Backup Method |
|-----------|-----|-----|---------------|
| PostgreSQL | 1 hour | 4 hours | WAL archiving + daily dumps |
| Neo4j | 24 hours | 8 hours | Daily Cypher exports |
| Redis | N/A | 30 min | AOF persistence |
| Secrets | N/A | 1 hour | External secret manager |

**Recovery Procedures:**

1. **Database Restore:**
   ```bash
   # PostgreSQL
   pg_restore -h postgres-service -U agentic_rag -d agentic_rag backup.dump

   # Neo4j
   cypher-shell -u neo4j -p $NEO4J_PASSWORD < backup.cypher
   ```

2. **Full Cluster Recovery:**
   ```bash
   # Apply all manifests
   kubectl apply -f k8s/

   # Restore databases
   kubectl exec -it postgres-0 -- pg_restore ...

   # Verify health
   kubectl get pods -n agentic-rag
   ```

---

## Troubleshooting

### Common Issues

#### Pod CrashLoopBackOff

```bash
# Check logs
kubectl logs -n agentic-rag deployment/backend --previous

# Common causes:
# - Missing secrets/configmaps
# - Database not ready
# - Invalid configuration
```

#### Database Connection Errors

```bash
# Check database pods
kubectl get pods -n agentic-rag -l app=postgres

# Test connectivity from backend
kubectl exec -it deployment/backend -n agentic-rag -- \
  python -c "import asyncpg; print('OK')"
```

#### High Memory Usage

```bash
# Check resource usage
kubectl top pods -n agentic-rag

# Adjust limits if needed
kubectl set resources deployment/backend \
  --limits=memory=16Gi -n agentic-rag
```

#### Slow Responses

```bash
# Check metrics
curl http://backend-service:8000/metrics | grep retrieval_latency

# Common causes:
# - Database slow queries
# - LLM provider latency
# - Insufficient replicas
```

### Useful Commands

```bash
# View all resources
kubectl get all -n agentic-rag

# Describe pod issues
kubectl describe pod <pod-name> -n agentic-rag

# Port-forward for debugging
kubectl port-forward svc/backend-service 8000:8000 -n agentic-rag

# Execute into pod
kubectl exec -it deployment/backend -n agentic-rag -- /bin/sh

# View events
kubectl get events -n agentic-rag --sort-by='.lastTimestamp'

# Rolling restart
kubectl rollout restart deployment/backend -n agentic-rag

# Check rollout status
kubectl rollout status deployment/backend -n agentic-rag
```

---

## See Also

- [Database Administration Guide](./database-administration.md) - Database configuration and maintenance
- [Observability Guide](./observability.md) - Prometheus metrics and alerting
- [Provider Configuration Guide](./provider-configuration.md) - LLM and embedding provider setup
- [CLI Installation Guide](./cli-installation.md) - CLI tool setup and usage
