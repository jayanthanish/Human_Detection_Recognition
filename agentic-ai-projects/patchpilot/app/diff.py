"""Parse unified diffs and anchor findings to changed lines."""

from pathlib import PurePosixPath
import re

MAX_PATCH = 2 * 1024 * 1024


class PatchError(ValueError):
    pass


def safe_path(path):
    parts = PurePosixPath(path).parts
    if (
        path.startswith("/")
        or ".." in parts
        or ".git" in parts
        or "\\" in path
        or not parts
    ):
        raise PatchError("Invalid patch file path.")
    return path


def parse_patch(text):
    if not isinstance(text, str) or len(text.encode()) > MAX_PATCH:
        raise PatchError("Patch must be 2 MB or smaller.")
    files = {}
    current = None
    new_line = None
    old_left = 0
    new_left = 0
    headers = []
    saw_old_header = False
    for line in text.splitlines():
        if line.startswith(chr(92) + " No newline"):
            continue
        if old_left or new_left:
            if current is None:
                raise PatchError("Hunk has no file header.")
            if line.startswith("+"):
                if not new_left:
                    raise PatchError("Hunk line count is inconsistent.")
                current["added"][new_line] = line[1:]
                current["context"][new_line] = line[1:]
                new_line += 1
                new_left -= 1
            elif line.startswith("-"):
                if not old_left:
                    raise PatchError("Hunk line count is inconsistent.")
                old_left -= 1
            elif line.startswith(" "):
                if not old_left or not new_left:
                    raise PatchError("Hunk line count is inconsistent.")
                current["context"][new_line] = line[1:]
                new_line += 1
                new_left -= 1
                old_left -= 1
            else:
                raise PatchError("The patch hunk is incomplete or malformed.")
            current["patch"] += "\n" + line
            continue
        if line.startswith("diff --git "):
            current = None
            headers = [line]
            saw_old_header = False
            continue
        if line.startswith("--- "):
            headers.append(line)
            saw_old_header = True
            continue
        if line.startswith("+++ ") and saw_old_header:
            path = line[4:].split("\t", 1)[0]
            saw_old_header = False
            if path == "/dev/null":
                current = None
                continue
            if path.startswith("b/"):
                path = path[2:]
            if path.startswith('"'):
                raise PatchError("Quoted patch paths are not supported.")
            path = safe_path(path)
            if path in files:
                raise PatchError("Duplicate file section in patch.")
            current = {
                "path": path,
                "patch": "\n".join(headers + [line]),
                "added": {},
                "context": {},
            }
            files[path] = current
            continue
        if line.startswith("@@") and current is not None:
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if not match:
                raise PatchError("Invalid hunk header.")
            old_left = int(match[2]) if match[2] is not None else 1
            new_left = int(match[4]) if match[4] is not None else 1
            new_line = int(match[3])
            current["patch"] += "\n" + line
    if old_left or new_left:
        raise PatchError("The patch ends before a hunk is complete.")
    if not files:
        raise PatchError("No readable changed-file hunks were found.")
    if len(files) > 40:
        raise PatchError("Review at most 40 changed files at a time.")
    return files


def validate_findings(data, files):
    if (
        not isinstance(data, dict)
        or set(data) != {"summary", "findings"}
        or not isinstance(data["summary"], str)
        or not isinstance(data["findings"], list)
    ):
        raise PatchError("Review must have summary and findings.")
    if len(data["findings"]) > 12:
        raise PatchError("Too many findings.")
    seen = set()
    for f in data["findings"]:
        keys = {
            "file",
            "line",
            "severity",
            "title",
            "description",
            "suggested_fix",
            "evidence",
        }
        if not isinstance(f, dict) or set(f) != keys:
            raise PatchError("Invalid finding shape.")
        if not all(isinstance(f[k], str) and f[k].strip() for k in keys - {"line"}):
            raise PatchError("Finding text fields must be nonempty strings.")
        if f["severity"] not in {"high", "medium", "low"}:
            raise PatchError("Unknown severity.")
        if (
            type(f["line"]) is not int
            or f["file"] not in files
            or f["line"] not in files[f["file"]]["added"]
        ):
            raise PatchError("Every finding must anchor to an added line in the patch.")
        if f["evidence"].strip() not in files[f["file"]]["added"][f["line"]]:
            raise PatchError("Evidence does not match the changed line.")
        key = (f["file"], f["line"], f["title"])
        if key in seen:
            raise PatchError("Duplicate finding.")
        seen.add(key)
    return data
