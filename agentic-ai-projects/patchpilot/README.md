# PatchPilot

### Agentic Code Review

A developer tool that uses an AI agent to inspect Git diffs and source context, then produces review findings anchored to added lines. The application validates each finding against the actual patch before saving a report.

## What it does

- Unified diff review and optional commit-to-commit review of a configured local Git repository.
- A bounded agent with list_changes, read_diff, and read_source tools.
- Head-commit source reads, exact changed-line evidence checks, severity validation, deduplication, and correction feedback.
- A browser review workspace with saved reports, tool activity, and Markdown export.

## Agent workflow

Git diff → changed-file/hunk parsing → agent diff/source inspection → finding validation → review report + tool audit

The agent has a finite tool-call and iteration budget. Tool errors are returned as observations so the model can correct its next action. The browser shows tool names, outcomes, and summaries, not hidden model reasoning.

## Run locally

Requires Python 3.12 and Ollama. Download the model before the first run:

```bash
ollama pull qwen3:8b
```

Install and start from this repository:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. API documentation is at http://127.0.0.1:8000/docs.

1. Click **Load sample diff**.
2. Click **Review changes**.
3. Inspect each finding’s file, new-line number, evidence, and suggested fix.
4. To review a local repository, set REPO_PATH before starting the server and enter base/head revisions.

## Configuration

Set environment variables before starting Uvicorn. For example, macOS/Linux uses `export CHAT_MODEL=qwen3:8b`; PowerShell uses `$env:CHAT_MODEL="qwen3:8b"`.

| Variable | Default | Purpose |
| --- | --- | --- |
| CHAT_MODEL | qwen3:8b | Tool-calling chat model |
| OLLAMA_URL | http://127.0.0.1:11434 | Your Ollama server |
| DATA_DIR | data | Persistent application data |
| REPO_PATH | unset | Optional local Git repository to inspect |

Inference calls go to OLLAMA_URL. With the default local URL, the application does not send prompts to a hosted model provider. Uploaded data is stored in DATA_DIR and is not committed to Git. The included sample data is fictional.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Coverage includes Unified hunk line mapping, path traversal rejection, added-line anchoring, fabricated evidence rejection, complete review API flow, history/export, and reading the reviewed Git commit even when the working tree has different contents.

Agent/API integration tests use scripted model responses and real application logic, SQLite, parsing, and validation. They verify orchestration and error handling; they are not a benchmark of an installed model’s answer quality. Live model inference needs your own running Ollama instance. GitHub Actions runs the same test suite on pushes and pull requests.

## Design decisions

The model chooses what to inspect, but never executes code or shell commands. Git access uses fixed argument arrays and resolved commit IDs. Findings must cite an added line in an inspected file. Source tools read from the reviewed commit, which avoids accidentally reviewing a different working-tree version.

## Repository layout

```text
app/
  agent.py          Tool schemas and bounded agent loop
  main.py           FastAPI endpoints and application wiring
  models.py         Ollama HTTP adapter
  static/           Browser UI
sample_data/        Reproducible example input
tests/              Unit and API integration tests
.github/workflows/  Continuous integration
```

## API surface

POST /api/review · POST /api/review/git · GET /api/reviews · GET /api/reviews/{id}/export

## Containers

A Dockerfile and compose.yaml are included. `docker compose up --build` exposes the app on 127.0.0.1:8000 and persists DATA_DIR in a named volume. The configured host Ollama URL must be reachable from inside the container; adjust OLLAMA_URL for your Docker network. Native Python is the simplest local setup. Container builds were not exercised in the build environment.

## Scope

Textual diffs only; deleted-only and binary files do not produce inline findings. Quoted Git paths are rejected. The application does not run tests, prove correctness, or post GitHub comments. A developer should verify findings. It is a single-user local tool; add authentication and repository authorization before network deployment.

## References

- [Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling)
- [Ollama chat API](https://docs.ollama.com/api/chat)
- [FastAPI](https://fastapi.tiangolo.com/)
