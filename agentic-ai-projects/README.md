# Agentic AI Applications

Three full stack applications focused on practical AI engineering.

| Project | Purpose | Agent tools |
| --- | --- | --- |
| [QueryDesk](querydesk) | Conversational data analysis with SQL, charts, and exports | Inspect dataset, execute read-only SQL |
| [PaperTrail](papertrail) | Multi-document research with hybrid retrieval and verified citations | List documents, search sources |
| [PatchPilot](patchpilot) | Code review with line-anchored findings and source context | List changes, read diffs, read source |

Each folder is an independent application with a FastAPI backend, browser UI, persistent storage, Docker configuration, sample inputs, and tests. See its README for setup.

The applications use local Ollama tool calling. The agent chooses tools while application code enforces call budgets, input validation, and output checks.

## Verification

43 tests passed across the three applications. Tests exercise real parsing, SQLite queries, retrieval, agent orchestration, and API flows with scripted model responses. Live model quality and container builds require validation on the target machine.

This collection lives on the `agentic-ai-applications` branch. The repository's original computer vision application remains on `main`.
