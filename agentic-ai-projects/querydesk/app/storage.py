"""CSV import and bounded, read-only SQL execution."""

import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import re
import sqlite3
import time
from uuid import uuid4
import sqlglot
from sqlglot import exp

MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 100_000
MAX_COLUMNS = 80
SAFE_FUNCTIONS = {
    "sum",
    "avg",
    "count",
    "min",
    "max",
    "round",
    "abs",
    "coalesce",
    "ifnull",
    "nullif",
    "lower",
    "upper",
    "length",
    "substr",
    "substring",
    "trim",
    "ltrim",
    "rtrim",
    "replace",
    "date",
    "datetime",
    "strftime",
    "julianday",
    "total",
    "iif",
    "like",
    "glob",
    "printf",
    "row_number",
    "rank",
    "dense_rank",
    "lag",
    "lead",
    "first_value",
    "last_value",
    "ntile",
    "percent_rank",
    "cume_dist",
}


class DataError(ValueError):
    pass


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, dataset_id):
        if not re.fullmatch(r"[0-9a-f]{32}", dataset_id):
            raise DataError("Invalid dataset identifier.")
        path = self.root / (dataset_id + ".sqlite")
        if not path.is_file():
            raise DataError("Dataset was not found.")
        return path

    def import_csv(self, name, content):
        if len(content) > MAX_BYTES:
            raise DataError("CSV must be 10 MB or smaller.")
        try:
            reader = csv.reader(io.StringIO(content.decode("utf-8-sig")), strict=True)
            original = next(reader)
            if not 1 <= len(original) <= MAX_COLUMNS:
                raise DataError("Use between 1 and 80 columns.")
            columns = []
            for i, header in enumerate(original):
                clean = (
                    re.sub(r"[^a-z0-9_]+", "_", header.strip().lower()).strip("_")
                    or f"column_{i+1}"
                )
                if clean[0].isdigit():
                    clean = "c_" + clean
                base = clean
                index = 2
                while clean in columns:
                    clean = f"{base}_{index}"
                    index += 1
                columns.append(clean)
            rows = []
            for row in reader:
                if not row:
                    continue
                if len(row) != len(columns):
                    raise DataError(
                        "A row has a different number of fields than the header."
                    )
                rows.append([cell.strip() if cell.strip() else None for cell in row])
                if len(rows) > MAX_ROWS:
                    raise DataError("CSV exceeds 100,000 rows.")
            if not rows:
                raise DataError("CSV contains no data rows.")
        except (UnicodeDecodeError, csv.Error, StopIteration) as exc:
            raise DataError(
                "Provide a valid UTF-8 CSV with a header and data rows."
            ) from exc
        types = []
        for i in range(len(columns)):
            cells = [r[i] for r in rows if r[i] is not None]
            # Preserve identifiers with significant leading zeros as text.
            numeric = bool(cells) and all(
                re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", v)
                and not re.fullmatch(r"0\d+", v)
                for v in cells
            )
            if numeric and all(math.isfinite(float(v)) for v in cells):
                types.append("REAL")
                for row in rows:
                    if row[i] is not None:
                        row[i] = float(row[i])
            else:
                types.append("TEXT")
        dataset_id = uuid4().hex
        schema = [
            {"name": c, "type": t, "original": o}
            for c, t, o in zip(columns, types, original)
        ]
        meta = {
            "id": dataset_id,
            "name": Path(name).name[:160],
            "rows": len(rows),
            "columns": schema,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self.root / (dataset_id + ".sqlite")
        try:
            with sqlite3.connect(path) as db:
                ddl = ", ".join('"' + c + '" ' + t for c, t in zip(columns, types))
                db.execute("CREATE TABLE data (" + ddl + ")")
                db.executemany(
                    "INSERT INTO data VALUES (" + ",".join("?" for _ in columns) + ")",
                    rows,
                )
                db.execute("CREATE TABLE metadata (json TEXT NOT NULL)")
                db.execute("INSERT INTO metadata VALUES (?)", (json.dumps(meta),))
                db.execute(
                    "CREATE TABLE history (id TEXT PRIMARY KEY, json TEXT NOT NULL)"
                )
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return meta

    def metadata(self, dataset_id):
        with sqlite3.connect(self.path(dataset_id)) as db:
            return json.loads(db.execute("SELECT json FROM metadata").fetchone()[0])

    def datasets(self):
        return sorted(
            [self.metadata(p.stem) for p in self.root.glob("*.sqlite")],
            key=lambda m: m["created_at"],
            reverse=True,
        )

    def execute(self, dataset_id, sql, row_limit=200, timeout=2.0):
        if not isinstance(sql, str) or len(sql) > 10_000:
            raise DataError("SQL is missing or too long.")
        try:
            statements = sqlglot.parse(sql, read="sqlite")
        except sqlglot.errors.ParseError as exc:
            raise DataError("The generated SQL could not be parsed.") from exc
        if len(statements) != 1 or not isinstance(
            statements[0], (exp.Select, exp.Union, exp.Intersect, exp.Except)
        ):
            raise DataError("Only a single SELECT query is allowed.")
        forbidden = {
            "insert",
            "update",
            "delete",
            "create",
            "drop",
            "alter",
            "command",
            "pragma",
            "attach",
            "detach",
            "transaction",
            "into",
            "copy",
        }
        if any(node.key in forbidden for node in statements[0].walk()):
            raise DataError("This query contains a forbidden operation.")
        clean = statements[0].sql(dialect="sqlite")
        start = time.monotonic()
        uri = self.path(dataset_id).as_uri() + "?mode=ro"
        db = sqlite3.connect(uri, uri=True)

        def authorize(action, arg1, arg2, database, trigger):
            if action in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE}:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and arg1 == "data" and database == "main":
                return sqlite3.SQLITE_OK
            if (
                action == sqlite3.SQLITE_FUNCTION
                and (arg2 or "").lower() in SAFE_FUNCTIONS
            ):
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        try:
            db.set_authorizer(authorize)
            db.set_progress_handler(
                lambda: int(time.monotonic() - start > timeout), 1000
            )
            cursor = db.execute(
                f"SELECT * FROM ({clean}) AS answer LIMIT {int(row_limit)+1}"
            )
            names = [c[0] for c in cursor.description]
            result = cursor.fetchall()
            if any(
                isinstance(cell, float) and not math.isfinite(cell)
                for row in result
                for cell in row
            ):
                raise DataError("The query produced a non-finite number.")
            return {
                "sql": clean,
                "columns": names,
                "rows": [list(r) for r in result[:row_limit]],
                "truncated": len(result) > row_limit,
                "duration_ms": round((time.monotonic() - start) * 1000, 2),
            }
        except sqlite3.Error as exc:
            raise DataError("Query rejected or could not execute: " + str(exc)) from exc
        finally:
            db.close()

    def save_history(self, dataset_id, record):
        with sqlite3.connect(self.path(dataset_id)) as db:
            db.execute(
                "INSERT INTO history VALUES (?,?)", (record["id"], json.dumps(record))
            )

    def history(self, dataset_id):
        with sqlite3.connect(self.path(dataset_id)) as db:
            return [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT json FROM history ORDER BY rowid DESC LIMIT 30"
                )
            ]
