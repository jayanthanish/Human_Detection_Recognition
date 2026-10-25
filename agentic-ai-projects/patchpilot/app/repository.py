"""Read Git objects without running repository scripts or diff helpers."""

import os
from pathlib import Path
import re
import subprocess
from .diff import PatchError, safe_path, MAX_PATCH


class Repository:
    def __init__(self, path):
        self.path = Path(path).resolve()

    def git(self, *args, limit=MAX_PATCH):
        try:
            result = subprocess.run(
                ["git", "-c", "core.hooksPath=/dev/null", "-C", str(self.path), *args],
                capture_output=True,
                timeout=10,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat"},
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PatchError("Git could not read the configured repository.") from exc
        if result.returncode:
            raise PatchError("Git could not resolve that revision or file.")
        if len(result.stdout) > limit:
            raise PatchError("Git output exceeds the review size limit.")
        return result.stdout.decode("utf-8", errors="replace")

    def commit(self, ref):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_./~^-]{0,150}", ref):
            raise PatchError("Invalid Git revision.")
        value = self.git(
            "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"
        ).strip()
        if not re.fullmatch(r"[0-9a-f]{40,64}", value):
            raise PatchError("Revision did not resolve to a commit.")
        return value

    def diff(self, base, head):
        base_sha = self.commit(base)
        head_sha = self.commit(head)
        return (
            self.git(
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--unified=8",
                base_sha,
                head_sha,
                "--",
            ),
            head_sha,
        )

    def source(self, commit, path, start, end):
        safe_path(path)
        if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
            raise PatchError("Invalid source revision.")
        if not 1 <= start <= end or end - start > 160:
            raise PatchError("Read at most 161 lines at a time.")
        text = self.git("show", commit + ":" + path, limit=400_000)
        return [
            {"line": i, "text": line}
            for i, line in enumerate(text.splitlines(), 1)
            if start <= i <= end
        ]
