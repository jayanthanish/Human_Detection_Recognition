# PaperTrail

### Agentic Document Research

A full stack research application that lets an AI agent search and compare documents through a bounded tool loop, then return answers with verified evidence quotes and page references.

## What it does

- PDF, Markdown, and text ingestion with document hashes, page-aware chunks, and persistent collections.
- Local embeddings plus SQLite FTS5 keyword search, fused with reciprocal rank fusion.
- An agent with list_documents and search_documents tools; searches can be narrowed to one source document.
- Exact quote validation, source-label validation, bounded citation repair, evidence cards, saved research history, and Markdown export.

## Agent workflow

Documents → page-aware chunks → local embeddings + FTS5 → agent catalog/search loop → citation validation → answer + page evidence

The agent has a finite tool-call and iteration budget. Tool errors are returned as observations so the model can correct its next action. The browser shows tool names, outcomes, and summaries, not hidden model reasoning.

## Run locally

Requires Python 3.12 and Ollama. Download the model before the first run:

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
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

1. Create a collection named **Policy comparison**.
2. Click **Add sample policies** to index the two fictional policy documents.
3. Ask “Compare annual leave and carryover at Northstar and Harbor.”
4. Review each quoted passage and export the research notes.

## Configuration

Set environment variables before starting Uvicorn. For example, macOS/Linux uses `export CHAT_MODEL=qwen3:8b`; PowerShell uses `$env:CHAT_MODEL="qwen3:8b"`.

| Variable | Default | Purpose |
| --- | --- | --- |
| CHAT_MODEL | qwen3:8b | Tool-calling chat model |
| OLLAMA_URL | http://127.0.0.1:11434 | Your Ollama server |
| DATA_DIR | data | Persistent application data |
| EMBED_MODEL | nomic-embed-text | Embedding model; keep fixed within a collection |

Inference calls go to OLLAMA_URL. With the default local URL, the application does not send prompts to a hosted model provider. Uploaded data is stored in DATA_DIR and is not committed to Git. The included sample data is fictional.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Coverage includes Real PDF text extraction, blank/scanned document rejection, chunk page numbers, collection isolation, duplicate uploads, hybrid retrieval, deletion cleanup, embedding dimension checks, fabricated citation rejection, source-label checks, and the upload → agent search → answer → history API flow.

Agent/API integration tests use scripted model responses and real application logic, SQLite, parsing, and validation. They verify orchestration and error handling; they are not a benchmark of an installed model’s answer quality. Live model inference needs your own running Ollama instance. GitHub Actions runs the same test suite on pushes and pull requests.

## Design decisions

Semantic and lexical retrieval serve different failure modes: the former finds related meaning and the latter preserves exact terms. Retrieval tools return stable source labels, document names, pages, and text. The application rejects citations that are absent from the retrieved evidence. The agent can revise an invalid citation within its iteration budget.

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

POST /api/collections · POST /api/collections/{id}/documents · POST /api/collections/{id}/ask · GET /api/collections/{id}/history

## Containers

A Dockerfile and compose.yaml are included. `docker compose up --build` exposes the app on 127.0.0.1:8000 and persists DATA_DIR in a named volume. The configured host Ollama URL must be reachable from inside the container; adjust OLLAMA_URL for your Docker network. Native Python is the simplest local setup. Container builds were not exercised in the build environment.

## Scope

PDF text extraction is supported; OCR is not included. Vectors are compared in-process, suitable for a local document collection rather than a large distributed index. Exact quotes establish source membership, not logical entailment. This is a single-user local workspace; add authentication and access controls before network deployment.

## References

- [Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling)
- [Ollama chat API](https://docs.ollama.com/api/chat)
- [FastAPI](https://fastapi.tiangolo.com/)
