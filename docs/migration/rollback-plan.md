# Rollback Plan

This plan describes how to revert to the previous release if the cutover
introduces a critical issue. It assumes the previous release tag is still
available in git and the previous images are still in the registry.

## Decision criteria

Roll back immediately if any of the following occur within the first hour
after cutover:

- Health endpoint returns non-200 for more than 2 consecutive minutes
- Login / auth flow is broken for all users
- Database migrations caused data loss or corruption
- Worker tasks fail at a rate above 10% with no clear fix

## Step 1 — Stop new traffic

1. Point the ingress / load balancer back to the previous API service.
2. Pause the worker queue consumption (stop the new worker pods).

## Step 2 — Roll back the API

```bash
# Docker Compose
docker compose -f deploy/docker/docker-compose.yml up -d --force-recreate api

# Kubernetes
kubectl rollout undo deployment/api-deployment
```

## Step 3 — Roll back the worker

```bash
# Docker Compose
docker compose -f deploy/docker/docker-compose.yml up -d --force-recreate worker

# Kubernetes
kubectl rollout undo deployment/worker-deployment
```

## Step 4 — Roll back the database

```bash
# Downgrade one migration step and restore the previous release tag.
./deploy/scripts/rollback.sh <previous-release-tag>
```

> **Caution**: `alembic downgrade` is destructive for the migration step it
> reverses. Only run this if the new migration caused the failure. If the
> failure is application-level (not schema-level), skip this step — the
> previous application can run against the migrated schema.

## Step 5 — Verify rollback

```bash
BASE_URL=https://api.example.com ./deploy/scripts/smoke-test.sh
```

The smoke test must pass against the rolled-back deployment.

## Step 6 — Post-rollback

- [ ] Confirm the previous release is serving traffic
- [ ] Confirm worker tasks are being processed
- [ ] Collect logs from the failed cutover for diagnosis
- [ ] File a follow-up issue describing the failure and the fix needed
- [ ] Do not re-attempt cutover until the root cause is fixed and tested

## Data safety notes

- The new schema is a superset of the old schema in most cases; downgrading
  may drop columns that the new version added. Back up the database before
  any downgrade.
- In-flight worker tasks from the failed cutover may be lost. Re-enqueue
  them after rollback if they are critical.
