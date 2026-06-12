#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

OVERALL_OK=true

# ── 1. Docker ─────────────────────────────────────────────────────────────────
echo "## DOCKER"

open -a Docker 2>/dev/null || true

MAX_RETRIES=30
RETRY_INTERVAL=2
attempt=0
compose_ok=false

while [ $attempt -lt $MAX_RETRIES ]; do
  if docker compose up -d; then
    compose_ok=true
    break
  fi
  attempt=$((attempt + 1))
  sleep $RETRY_INTERVAL
done

if [ "$compose_ok" = "true" ]; then
  echo "COMPOSE_UP: ok (after $((attempt * RETRY_INTERVAL))s)"
else
  echo "TIMEOUT: docker compose up did not succeed after $((MAX_RETRIES * RETRY_INTERVAL))s"
  echo ""
  echo "## SUMMARY"
  echo "FAILED: docker timeout"
  exit 1
fi

# ── 2. Health ─────────────────────────────────────────────────────────────────
echo ""
echo "## HEALTH"

echo "Waiting 25s for services to initialize..."
sleep 25

MAX_HEALTH_RETRIES=60
HEALTH_INTERVAL=5
attempt=0
health_ok=false

while [ $attempt -lt $MAX_HEALTH_RETRIES ]; do
  status="$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || true)"
  if [ "$status" = "200" ]; then
    health_ok=true
    break
  fi
  attempt=$((attempt + 1))
  sleep $HEALTH_INTERVAL
done

if [ "$health_ok" = "true" ]; then
  echo "STATUS: 200 (after $((25 + attempt * HEALTH_INTERVAL))s total)"
else
  echo "TIMEOUT: /health did not return 200 after $((25 + MAX_HEALTH_RETRIES * HEALTH_INTERVAL))s"
  echo ""
  echo "## SUMMARY"
  echo "FAILED: health timeout"
  exit 1
fi

# ── 3. Tests ──────────────────────────────────────────────────────────────────
echo ""
echo "## TESTS"

pytest_output="$(python -m pytest tests/ -q 2>&1)" || true
pytest_exit=$?

echo "$pytest_output"

if [ $pytest_exit -eq 0 ]; then
  echo "RESULT: PASSED"
else
  echo "RESULT: FAILED (exit $pytest_exit)"
  OVERALL_OK=false
fi

# ── 4. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "## SUMMARY"

if [ "$OVERALL_OK" = "true" ]; then
  echo "OK"
  exit 0
else
  echo "FAILED: tests"
  exit 1
fi
