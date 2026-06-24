---
name: initialize-project
description: >
  Use this skill when the user wants to get the project up and running —
  whether they say "initialize project", "start the stack", "spin up the
  app", or "get things working locally".
allowed-tools: Bash
---

Run the initialization script from the project root:

```bash
bash ${CLAUDE_SKILL_DIR}/scripts/init-project.sh
```

The script prints structured sections prefixed with `## SECTION`. Parse these to build your response.

## How to respond

**On `## SUMMARY` line `OK`:**

Print exactly:

```
App is up and running. All tests passed.

→ http://localhost:8000

To evaluate agent quality: python -m evals.run_evals  (needs GOOGLE_API_KEY — uses Gemini quota)
```

**On `## SUMMARY` line `FAILED: docker timeout`:**

Tell the user: "Docker Desktop didn't finish starting in time. Check the whale icon in your menu bar — if it's still animating, wait for it to stop, then run `/initialize-project` again."

**On `## SUMMARY` line `FAILED: health timeout`:**

Tell the user: "The app didn't become reachable after ~5 minutes. Run `docker compose logs api` to check for startup errors. On first run this may be caused by a slow HuggingFace model download — try again."

**On `## SUMMARY` line `FAILED: tests`:**

Tell the user: "The stack is up but the test suite has failures. Run `python -m pytest tests/ -v` for detailed output." Include the failed test names from the `## TESTS` section if present.

**On any other non-zero exit or unexpected output:**

Report which `## SECTION` was the last one printed before the failure and paste the raw output from that section so the user has the context to debug.
