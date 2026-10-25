import json
from pathlib import Path
import subprocess
import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app.diff import parse_patch, validate_findings, PatchError
from app.repository import Repository
from app.agent import review
from app.models import ModelError

PATCH = (Path(__file__).parent.parent / "sample_data/checkout.patch").read_text()
FINDING = {
    "file": "checkout.py",
    "line": 2,
    "severity": "medium",
    "title": "Reject discounts above 100%",
    "description": "A 150% discount makes the returned total negative, whereas the old validation rejected it.",
    "suggested_fix": "Keep the upper bound check.",
    "evidence": "if discount_percent < 0:",
}


class ReviewModel:
    model = "test"

    def __init__(self):
        self.step = 0

    def tool_chat(self, messages, tools):
        self.step += 1
        if self.step == 1:
            return {
                "content": "",
                "tool_calls": [{"function": {"name": "list_changes", "arguments": {}}}],
            }
        if self.step == 2:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_diff",
                            "arguments": {"file": "checkout.py"},
                        }
                    }
                ],
            }
        return {
            "content": json.dumps(
                {
                    "summary": "The change removes the upper discount limit.",
                    "findings": [FINDING],
                }
            )
        }


def test_hunk_line_numbers():
    f = parse_patch(PATCH)["checkout.py"]
    assert f["added"] == {2: "    if discount_percent < 0:"}
    assert f["context"][5] == "    return total - discount"


@pytest.mark.parametrize("path", ["../secret.py", "/etc/passwd", ".git/config"])
def test_path_traversal_rejected(path):
    patch = PATCH.replace("b/checkout.py", "b/" + path)
    with pytest.raises(PatchError):
        parse_patch(patch)


def test_findings_must_anchor_to_added_line():
    files = parse_patch(PATCH)
    assert validate_findings({"summary": "Review", "findings": [FINDING]}, files)
    with pytest.raises(PatchError):
        validate_findings(
            {"summary": "Review", "findings": [{**FINDING, "line": 3}]}, files
        )


def test_fabricated_evidence_rejected():
    with pytest.raises(PatchError):
        validate_findings(
            {
                "summary": "Review",
                "findings": [{**FINDING, "evidence": "something invented"}],
            },
            parse_patch(PATCH),
        )


def test_agent_tools_and_validated_findings():
    result = review(parse_patch(PATCH), ReviewModel())
    assert result["findings"][0]["line"] == 2
    assert len(result["trace"]) == 2


def test_api_review_history_and_export(tmp_path):
    client = TestClient(create_app(tmp_path, ReviewModel()))
    r = client.post("/api/review", json={"patch": PATCH})
    assert r.status_code == 200, r.text
    assert len(client.get("/api/reviews").json()) == 1
    assert (
        "Reject discounts"
        in client.get("/api/reviews/" + r.json()["id"] + "/export").text
    )
    assert client.post("/api/review/git", json={"base": "HEAD~1"}).status_code == 422
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_git_reader_uses_reviewed_commit(tmp_path):
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()

    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    (tmp_path / "code.py").write_text("value = 1\n")
    git("add", ".")
    git("commit", "-m", "base")
    (tmp_path / "code.py").write_text("value = 2\n")
    git("add", ".")
    git("commit", "-m", "change")
    repo = Repository(tmp_path)
    diff, head = repo.diff("HEAD~1", "HEAD")
    assert "+value = 2" in diff
    (tmp_path / "code.py").write_text("value = 999\n")
    assert repo.source(head, "code.py", 1, 1) == [{"line": 1, "text": "value = 2"}]
    with pytest.raises(PatchError):
        repo.commit("--help")


def test_no_newline_markers_and_header_like_code():
    patch = (
        "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n"
        + chr(92)
        + " No newline at end of file\n+++ new\n"
        + chr(92)
        + " No newline at end of file\n"
    )
    assert parse_patch(patch)["x.py"]["added"][1] == "++ new"


def test_truncated_hunk_rejected():
    with pytest.raises(PatchError):
        parse_patch(PATCH.rsplit("\n", 2)[0])
