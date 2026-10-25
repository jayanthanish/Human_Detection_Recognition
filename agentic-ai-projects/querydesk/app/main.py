from datetime import datetime, timezone
import csv
import io
import json
import os
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .storage import Store, DataError, MAX_BYTES
from .models import Ollama, ModelError
from .agent import run_agent


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1500)
    previous_question: str | None = Field(default=None, max_length=1500)
    previous_sql: str | None = Field(default=None, max_length=10000)


def create_app(data_dir=None, model=None):
    app = FastAPI(title="QueryDesk", version="1.0.0")
    store = Store(data_dir or os.getenv("DATA_DIR", "data"))
    ai = model or Ollama()
    app.state.store = store

    @app.exception_handler(DataError)
    def data_error(request, exc):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(ModelError)
    def model_error(request, exc):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/api/health")
    def health():
        return {"status": "ok", "chat_model": getattr(ai, "model", "injected")}

    @app.get("/api/datasets")
    def datasets():
        return store.datasets()

    @app.post("/api/datasets")
    def upload(file: UploadFile = File(...)):
        if not (file.filename or "").lower().endswith(".csv"):
            raise DataError("Upload a CSV file.")
        content = file.file.read(MAX_BYTES + 1)
        return store.import_csv(file.filename, content)

    @app.post("/api/sample")
    def sample():
        path = Path(__file__).parent.parent / "sample_data" / "sales.csv"
        return store.import_csv(path.name, path.read_bytes())

    @app.get("/api/datasets/{dataset_id}")
    def dataset(dataset_id: str):
        meta = store.metadata(dataset_id)
        return {
            **meta,
            "preview": store.execute(dataset_id, "SELECT * FROM data LIMIT 8"),
        }

    @app.get("/api/datasets/{dataset_id}/history")
    def history(dataset_id: str):
        return store.history(dataset_id)

    @app.post("/api/datasets/{dataset_id}/ask")
    def ask(dataset_id: str, request: Question):
        store.metadata(dataset_id)
        result = run_agent(
            store,
            dataset_id,
            request.question,
            ai,
            (
                {"question": request.previous_question, "sql": request.previous_sql}
                if request.previous_question
                else None
            ),
        )
        if result["status"] == "needs_context":
            return result
        record = {
            "id": uuid4().hex,
            "question": request.question,
            "explanation": result["explanation"],
            **result,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        store.save_history(dataset_id, record)
        return {"status": "answered", **record}

    @app.get("/api/datasets/{dataset_id}/history/{query_id}/export")
    def export(dataset_id: str, query_id: str):
        record = next(
            (r for r in store.history(dataset_id) if r["id"] == query_id), None
        )
        if not record:
            raise HTTPException(404, "Query was not found in recent history.")
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(record["columns"])
        for row in record["rows"]:
            # Spreadsheet formula escaping for exported string values.
            writer.writerow(
                [
                    (
                        "'" + v
                        if isinstance(v, str)
                        and v.startswith(("=", "+", "-", "@", "\t", "\r"))
                        else v
                    )
                    for v in row
                ]
            )
        return Response(
            output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=query-results.csv"},
        )

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def home():
        return FileResponse(static / "index.html")

    return app


app = create_app()
