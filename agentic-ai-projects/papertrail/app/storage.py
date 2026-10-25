"""SQLite document collections, FTS5, vectors, and durable answer history."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sqlite3
from uuid import uuid4
from .documents import DocumentError


def now():
    return datetime.now(timezone.utc).isoformat()


def cosine(a, b):
    if len(a) != len(b):
        raise DocumentError(
            "Embedding dimensions changed. Use the original embedding model."
        )
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / denom if denom else 0


class Store:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(
                """CREATE TABLE IF NOT EXISTS collections (id TEXT PRIMARY KEY,name TEXT NOT NULL,embedding_model TEXT NOT NULL,created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY,collection_id TEXT NOT NULL REFERENCES collections(id),name TEXT NOT NULL,sha256 TEXT NOT NULL,pages INTEGER NOT NULL,created_at TEXT NOT NULL,UNIQUE(collection_id,sha256));
            CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY,document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,collection_id TEXT NOT NULL,page INTEGER NOT NULL,text TEXT NOT NULL,vector TEXT NOT NULL);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search USING fts5(id UNINDEXED,collection_id UNINDEXED,text);
            CREATE TABLE IF NOT EXISTS answers (id TEXT PRIMARY KEY,collection_id TEXT NOT NULL REFERENCES collections(id),json TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS delete_chunk_search AFTER DELETE ON chunks BEGIN DELETE FROM chunk_search WHERE id=old.id; END;"""
            )

    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def create_collection(self, name, model):
        record = {
            "id": uuid4().hex,
            "name": name,
            "embedding_model": model,
            "created_at": now(),
        }
        with self.connect() as db:
            db.execute(
                "INSERT INTO collections VALUES (:id,:name,:embedding_model,:created_at)",
                record,
            )
        return record

    def collections(self):
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT c.*,COUNT(d.id) AS document_count FROM collections c LEFT JOIN documents d ON d.collection_id=c.id GROUP BY c.id ORDER BY c.created_at DESC"
                )
            ]

    def collection(self, collection_id):
        with self.connect() as db:
            r = db.execute(
                "SELECT * FROM collections WHERE id=?", (collection_id,)
            ).fetchone()
        if not r:
            raise DocumentError("Collection was not found.")
        return dict(r)

    def documents(self, collection_id):
        self.collection(collection_id)
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM documents WHERE collection_id=? ORDER BY created_at DESC",
                    (collection_id,),
                )
            ]

    def add_document(self, collection_id, document, chunks, vectors):
        self.collection(collection_id)
        if len(chunks) != len(vectors) or not chunks:
            raise DocumentError("Every chunk needs an embedding.")
        if len(vectors[0]) < 1 or any(
            len(v) != len(vectors[0])
            or not all(type(n) in (int, float) and math.isfinite(n) for n in v)
            or not any(v)
            for v in vectors
        ):
            raise DocumentError("Invalid document embeddings.")
        record = {
            "id": uuid4().hex,
            "collection_id": collection_id,
            "name": document["name"],
            "sha256": document["sha256"],
            "pages": len(document["pages"]),
            "created_at": now(),
        }
        try:
            with self.connect() as db:
                existing = db.execute(
                    "SELECT vector FROM chunks WHERE collection_id=? LIMIT 1",
                    (collection_id,),
                ).fetchone()
                if existing and len(json.loads(existing[0])) != len(vectors[0]):
                    raise DocumentError(
                        "Embedding dimensions differ from this collection."
                    )
                db.execute(
                    "INSERT INTO documents VALUES (:id,:collection_id,:name,:sha256,:pages,:created_at)",
                    record,
                )
                for chunk, vector in zip(chunks, vectors):
                    chunk_id = uuid4().hex
                    db.execute(
                        "INSERT INTO chunks VALUES (?,?,?,?,?,?)",
                        (
                            chunk_id,
                            record["id"],
                            collection_id,
                            chunk["page"],
                            chunk["text"],
                            json.dumps(vector),
                        ),
                    )
                    db.execute(
                        "INSERT INTO chunk_search VALUES (?,?,?)",
                        (chunk_id, collection_id, chunk["text"]),
                    )
        except sqlite3.IntegrityError as exc:
            raise DocumentError("This document is already in the collection.") from exc
        return {**record, "chunk_count": len(chunks)}

    def delete_document(self, collection_id, document_id):
        self.collection(collection_id)
        with self.connect() as db:
            result = db.execute(
                "DELETE FROM documents WHERE id=? AND collection_id=?",
                (document_id, collection_id),
            )
            if not result.rowcount:
                raise DocumentError("Document was not found.")

    def retrieve(
        self, collection_id, question, query_vector, top_k=5, document_id=None
    ):
        self.collection(collection_id)
        with self.connect() as db:
            rows = [
                dict(r)
                for r in db.execute(
                    "SELECT c.*,d.name AS document FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.collection_id=?",
                    (collection_id,),
                )
            ]
            if document_id:
                rows = [r for r in rows if r["document_id"] == document_id]
            terms = re.findall(r"\w+", question.lower())[:40]
            fts_query = " OR ".join('"' + t + '"' for t in terms)
            if fts_query:
                if document_id:
                    lexical = [
                        r["id"]
                        for r in db.execute(
                            "SELECT id FROM chunk_search WHERE chunk_search MATCH ? AND collection_id=? AND id IN (SELECT id FROM chunks WHERE document_id=?) ORDER BY bm25(chunk_search) LIMIT 20",
                            (fts_query, collection_id, document_id),
                        )
                    ]
                else:
                    lexical = [
                        r["id"]
                        for r in db.execute(
                            "SELECT id FROM chunk_search WHERE chunk_search MATCH ? AND collection_id=? ORDER BY bm25(chunk_search) LIMIT 20",
                            (fts_query, collection_id),
                        )
                    ]
            else:
                lexical = []
        allowed = {r["id"] for r in rows}
        lexical = [cid for cid in lexical if cid in allowed]
        vector_scores = {
            r["id"]: cosine(query_vector, json.loads(r["vector"])) for r in rows
        }
        semantic = sorted(
            [cid for cid, score in vector_scores.items() if score >= 0.25],
            key=lambda cid: vector_scores[cid],
            reverse=True,
        )[:20]
        fused = {}
        for ranking in (lexical, semantic):
            for rank, cid in enumerate(ranking, 1):
                fused[cid] = fused.get(cid, 0) + 1 / (60 + rank)
        indexed = {r["id"]: r for r in rows}
        return [
            {
                "id": cid,
                "document": indexed[cid]["document"],
                "document_id": indexed[cid]["document_id"],
                "page": indexed[cid]["page"],
                "text": indexed[cid]["text"],
                "score": round(fused[cid], 6),
                "similarity": round(vector_scores[cid], 4),
            }
            for cid in sorted(fused, key=lambda cid: (-fused[cid], cid))[:top_k]
        ]

    def save_answer(self, collection_id, record):
        with self.connect() as db:
            db.execute(
                "INSERT INTO answers VALUES (?,?,?)",
                (record["id"], collection_id, json.dumps(record)),
            )

    def history(self, collection_id):
        self.collection(collection_id)
        with self.connect() as db:
            return [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT json FROM answers WHERE collection_id=? ORDER BY rowid DESC LIMIT 30",
                    (collection_id,),
                )
            ]
