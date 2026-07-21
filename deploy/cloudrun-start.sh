#!/bin/bash
set -e
python mcp_server/server.py &
for i in $(seq 1 60); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8001/mcp)" != "000" ] && break
  sleep 1
done
exec uvicorn agent.server:app --host 0.0.0.0 --port "${PORT:-8000}"
