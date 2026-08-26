---
name: deploy-cloud-run
description: >
  Use this skill whenever the user wants to deploy or redeploy the
  jewish-book-guide project to its live Cloud Run service — phrases like
  "deploy this", "redeploy to cloud run", "ship this to prod", "push the
  update live", or "deploy the latest changes."
allowed-tools: Bash, Read
---

Deploying redeploys the live Cloud Run service, which serves real traffic and talks to the production Supabase database — so this is not a purely local, reversible action. Always show the user the exact command you're about to run and get their go-ahead before executing it — don't run it silently as part of a larger task.

## Why this needs care

Two things about this project's deploy make it easy to get wrong:

1. **`--set-env-vars` and `--set-secrets` fully replace the existing set — they are not additive.** Cloud Run revisions otherwise inherit whatever you don't explicitly touch from the previous revision, so on a routine redeploy where nothing about env vars/secrets is changing, just omit `--set-env-vars`/`--set-secrets` entirely rather than re-listing them. When you do need to add or change one, prefer `--update-secrets`/`--update-env-vars` (merges) over the `--set-*` form. Only reach for `--set-*` if you deliberately want to reset the whole set, and in that case run `gcloud run services describe jewish-book-guide --region=australia-southeast1` first to see what's currently live so nothing gets dropped by accident.

2. **The Dockerfile's own `CMD` does not start the books MCP server.** The deploy command overrides it with `--command deploy/cloudrun-start.sh`, which starts the MCP server before the API server. If a deploy ever runs without that override, `load_books_tools()` will silently swallow the failure and the agent will come up without book tools — so never drop `--command=deploy/cloudrun-start.sh` from the command, even though other unspecified flags are safe to omit.

## The deploy command

**Routine redeploy** (code change only, no new/changed secrets or env vars):

```bash
gcloud run deploy jewish-book-guide \
  --source . --project=jewish-book-guide --region=australia-southeast1 \
  --command=deploy/cloudrun-start.sh \
  --memory=2Gi --cpu=2 --concurrency=10 --max-instances=20 --timeout=300 \
  --allow-unauthenticated --quiet
```

**Adding or changing a secret or env var** — merge it in with `--update-secrets`/`--update-env-vars` instead of reconstructing the full list:

```bash
gcloud run deploy jewish-book-guide \
  --source . --project=jewish-book-guide --region=australia-southeast1 \
  --command=deploy/cloudrun-start.sh \
  --update-secrets="NOTION_API_KEY=NOTION_API_KEY:latest" \
  --memory=2Gi --cpu=2 --concurrency=10 --max-instances=20 --timeout=300 \
  --allow-unauthenticated --quiet
```

For reference, what's currently live (only needed if you're doing a full `--set-*` reset):
- env vars: `LANGSMITH_TRACING=true,LANGSMITH_ENDPOINT=https://api.smith.langchain.com,ALLOWED_ORIGINS=https://jewish-book-guide-887998576030.australia-southeast1.run.app,LANGSMITH_PROJECT=Jewish Book Guide`
- secrets: `GOOGLE_API_KEY=GOOGLE_API_KEY:latest,YOUTUBE_API_KEY=YOUTUBE_API_KEY:latest,LANGCHAIN_API_KEY=LANGCHAIN_API_KEY:latest,DATABASE_URL=DATABASE_URL:latest`

## After deploying

- `GET /health` is liveness only (no DB check) — it will pass even if the database is unreachable. To actually confirm the new revision is healthy, check `GET /ready`, which runs `SELECT 1` against the database with bounded timeouts.
- A Cloud Scheduler job pings `/ready` every ~5 minutes to keep Cloud Run and the free-tier Supabase instance from going idle. If the user reports cold-start latency despite that, `min-instances=1` is a paid option that removes cold starts entirely — mention it as an option, don't set it unasked.

## If something looks off before you run it

If the user's request implies a change to the deploy shape itself (new secret, new env var, different resources), work out the right `--update-*`/`--set-*` flags with them rather than guessing — then run `gcloud run services describe` afterward to confirm the change landed as expected.
