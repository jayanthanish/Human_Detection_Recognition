import json
import os
from pathlib import Path
import sqlite3
from uuid import uuid4
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .agent import review
from .diff import parse_patch, PatchError, MAX_PATCH
from .repository import Repository
from .models import Ollama, ModelError


class ReviewRequest(BaseModel):
    patch: str = Field(min_length=1, max_length=MAX_PATCH)


class GitReview(BaseModel):
    base: str = Field(min_length=1, max_length=151)
    head: str = Field(default="HEAD", max_length=151)


def create_app(data_dir=None, model=None, repository=None):
    app = FastAPI(title="PatchPilot", version="1.0.0")
    ai = model or Ollama()
    repo = repository or (
        Repository(os.environ["REPO_PATH"]) if os.getenv("REPO_PATH") else None
    )
    path = Path(data_dir or os.getenv("DATA_DIR", "data")).resolve() / "reviews.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS reviews (id TEXT PRIMARY KEY,json TEXT NOT NULL)"
        )

    @app.exception_handler(PatchError)
    def patch_error(request, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(ModelError)
    def model_error(request, exc):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    def save(files, result):
        result = {
            "id": uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": list(files),
            **result,
        }
        with sqlite3.connect(path) as db:
            db.execute(
                "INSERT INTO reviews VALUES (?,?)", (result["id"], json.dumps(result))
            )
        return result

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "chat_model": getattr(ai, "model", "injected"),
            "git_enabled": repo is not None,
        }

    @app.get("/api/sample")
    def sample():
        return {
            "patch": (
                Path(__file__).parent.parent / "sample_data" / "checkout.patch"
            ).read_text()
        }

    @app.post("/api/review")
    def review_patch(request: ReviewRequest):
        files = parse_patch(request.patch)
        return save(files, review(files, ai))

    @app.post("/api/review/git")
    def review_git(request: GitReview):
        if not repo:
            raise PatchError("Set REPO_PATH on the server to enable Git review.")
        patch, commit = repo.diff(request.base, request.head)
        files = parse_patch(patch)
        return save(files, review(files, ai, repo, commit))

    @app.get("/api/reviews")
    def history():
        with sqlite3.connect(path) as db:
            return [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT json FROM reviews ORDER BY rowid DESC LIMIT 30"
                )
            ]

    @app.get("/api/reviews/{review_id}/export")
    def export(review_id: str):
        with sqlite3.connect(path) as db:
            row = db.execute(
                "SELECT json FROM reviews WHERE id=?", (review_id,)
            ).fetchone()
        if not row:
            return JSONResponse(
                status_code=404, content={"detail": "Review was not found."}
            )
        result = json.loads(row[0])
        lines = ["# Code review", "", result["summary"]]
        for f in result["findings"]:
            lines.extend(
                [
                    "",
                    f"## {f['severity'].upper()} — {f['title']}",
                    f"{f['file']}:{f['line']}",
                    "",
                    f["description"],
                    "",
                    "Suggested fix: " + f["suggested_fix"],
                    "",
                    "```",
                    f["evidence"],
                    "```",
                ]
            )
        return Response(
            "\n".join(lines),
            media_type="text/markdown",
            headers={"Content-Disposition": "attachment; filename=review.md"},
        )

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def home():
        return FileResponse(static / "index.html")

    return app


app = create_app()
