# Cutover Runbook

This runbook describes the step-by-step procedure for switching production
traffic from the legacy backend to the new backend. It is written to be
executed by an operator with access to the deployment environment.

## Prerequisites

- Docker + Docker Compose (or a Kubernetes cluster with `kubectl`)
- `uv` (Python package manager) for local tooling
- Access to the target database and Redis instance
- A release tag from the previous deployment (for rollback)

## Phase 0 — Preflight

```bash
# Verify all runtime prerequisites are present and reachable.
./deploy/scripts/preflight.sh
```

The preflight script checks: Python >= 3.11, `uv`, Docker, `psql`,
`redis-cli`, database reachability, and Redis reachability. It must exit 0
before proceeding.

## Phase 1 — Build images

```bash
# API image
docker build -f deploy/docker/Dockerfile -t knowledge-be-api:<tag> .

# Worker image
docker build -f deploy/docker/Dockerfile --target worker -t knowledge-be-worker:<tag> .
```

## Phase 2 — Migrate the database

```bash
# Apply all pending migrations and tag the release.
./deploy/scripts/migrate-and-tag.sh <release-tag>
```

This runs `alembic upgrade head`, verifies a single migration head, and
tags the release in git. The migration chain is linear; a multi-head state
aborts the cutover.

## Phase 3 — Deploy the API

### Docker Compose

```bash
docker compose -f deploy/docker/docker-compose.yml up -d api
```

### Kubernetes

```bash
kubectl apply -f deploy/k8s/api-configmap.yaml
kubectl apply -f deploy/k8s/api-secret.yaml.example   # after filling secrets
kubectl apply -f deploy/k8s/api-deployment.yaml
kubectl apply -f deploy/k8s/api-service.yaml
kubectl apply -f deploy/k8s/api-ingress.yaml
```

## Phase 4 — Deploy the worker

```bash
# Docker Compose
docker compose -f deploy/docker/docker-compose.yml up -d worker

# Kubernetes
kubectl apply -f deploy/k8s/worker-configmap.yaml
kubectl apply -f deploy/k8s/worker-secret.yaml.example  # after filling secrets
kubectl apply -f deploy/k8s/worker-deployment.yaml
kubectl apply -f deploy/k8s/worker-service.yaml
```

## Phase 5 — Smoke test

```bash
BASE_URL=https://api.example.com ./deploy/scripts/smoke-test.sh
```

The smoke test verifies: health endpoint, OpenAPI schema (>= 50 paths), and
auth validation (422 on empty login body). All three must pass.

## Phase 6 — Switch traffic

1. Update the ingress / load balancer to point at the new API service.
2. Verify the worker is consuming from the same Redis queue the API enqueues
   into.
3. Monitor logs for the first 15 minutes:
   - API: `kubectl logs -f deploy/api-<pod>` or `docker compose logs -f api`
   - Worker: `kubectl logs -f deploy/worker-<pod>` or `docker compose logs -f worker`

## Phase 7 — Verify

- [ ] Health endpoint returns 200
- [ ] Login flow works end-to-end
- [ ] A knowledge-base search returns results
- [ ] A document upload triggers a worker task that completes
- [ ] Chat streaming responds with SSE events

## Post-cutover

- Keep the previous release tag available for at least 7 days.
- Record the cutover timestamp and release tag in the deployment log.
- If any critical issue is found, follow the rollback plan immediately.
