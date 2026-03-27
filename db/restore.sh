#!/bin/bash
set -e
pg_restore --no-owner --no-acl -U "$POSTGRES_USER" -d "$POSTGRES_DB" /docker-entrypoint-initdb.d/books.dump || true
