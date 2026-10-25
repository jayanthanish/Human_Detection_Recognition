import io
import json
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from app.documents import parse_document, make_chunks, DocumentError
from app.storage import Store, cosine
from app.main import create_app, checked_answer
from app.models import ModelError


class ResearchModel:
    model = "test"
    embedding_model = "test-embed"

    def __init__(self):
        self.step = 0

    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def tool_chat(self, messages, tools):
        self.step += 1
        if self.step == 1:
            return {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "list_documents", "arguments": {}}}
                ],
            }
        if self.step == 2:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_documents",
                            "arguments": {"query": "annual leave"},
                        }
                    }
                ],
            }
        passages = json.loads(
            [
                m["content"]
                for m in messages
                if m["role"] == "tool" and m["tool_name"] == "search_documents"
            ][-1]
        )
        p = passages[0]
        return {
            "content": json.dumps(
                {
                    "answer": "See the documented policy [" + p["source_id"] + "].",
                    "abstained": False,
                    "citations": [{"source_id": p["source_id"], "quote": p["text"]}],
                }
            )
        }


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "research.sqlite")


def test_text_parsing_and_chunking():
    doc = parse_document("example.md", b"Hello\nworld.")
    assert doc["pages"] == [(1, "Hello world.")]
    assert make_chunks([(2, "one two three four five")], 3, 1) == [
        {"page": 2, "text": "one two three"},
        {"page": 2, "text": "three four five"},
    ]


def test_real_pdf_text_and_page_numbers():
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 200 Td (Annual leave is 20 days.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    out = io.BytesIO()
    writer.write(out)
    doc = parse_document("policy.pdf", out.getvalue())
    assert doc["pages"][0][0] == 1 and "20 days" in doc["pages"][0][1]


def test_blank_pdf_rejected():
    writer = PdfWriter()
    writer.add_blank_page(width=10, height=10)
    out = io.BytesIO()
    writer.write(out)
    with pytest.raises(DocumentError, match="No readable"):
        parse_document("blank.pdf", out.getvalue())


@pytest.mark.parametrize(
    "name,content",
    [("a.exe", b"abc"), ("a.txt", b""), ("a.txt", b"\xff"), ("bad.pdf", b"not a pdf")],
)
def test_bad_documents(name, content):
    with pytest.raises(DocumentError):
        parse_document(name, content)


def test_hybrid_retrieval_and_collection_isolation(store):
    c = store.create_collection("A", "test")
    other = store.create_collection("B", "test")
    doc = parse_document("leave.txt", b"Annual leave is 20 days.")
    chunks = make_chunks(doc["pages"])
    record = store.add_document(c["id"], doc, chunks, [[1.0, 0.0]])
    hits = store.retrieve(c["id"], "annual leave", [1.0, 0.0])
    assert hits[0]["document"] == "leave.txt"
    assert store.retrieve(other["id"], "annual leave", [1.0, 0.0]) == []
    with pytest.raises(DocumentError):
        store.add_document(c["id"], doc, chunks, [[1.0, 0.0]])
    store.delete_document(c["id"], record["id"])
    assert store.retrieve(c["id"], "annual leave", [1.0, 0.0]) == []


def test_embedding_dimension_change(store):
    c = store.create_collection("A", "test")
    doc = parse_document("a.txt", b"one two")
    store.add_document(c["id"], doc, make_chunks(doc["pages"]), [[1.0, 0.0]])
    with pytest.raises(DocumentError):
        store.retrieve(c["id"], "one", [1.0, 0.0, 0.0])


def test_validated_answer_and_bad_citations():
    sources = [{"source_id": "S1", "text": "Annual leave is 20 days."}]
    valid = {
        "answer": "20 days [S1].",
        "abstained": False,
        "citations": [{"source_id": "S1", "quote": "20 days"}],
    }
    assert checked_answer(json.dumps(valid), sources)["answer"] == "20 days [S1]."
    valid["citations"][0]["quote"] = "40 days"
    with pytest.raises(ModelError):
        checked_answer(json.dumps(valid), sources)


def test_unknown_source_label_rejected():
    with pytest.raises(ModelError):
        checked_answer(
            json.dumps(
                {
                    "answer": "20 days [S99].",
                    "abstained": False,
                    "citations": [{"source_id": "S1", "quote": "20 days"}],
                }
            ),
            [{"source_id": "S1", "text": "20 days"}],
        )


def test_api_document_research_and_history(tmp_path):
    client = TestClient(create_app(tmp_path, ResearchModel()))
    c = client.post("/api/collections", json={"name": "Policies"}).json()
    cid = c["id"]
    r = client.post(
        "/api/collections/" + cid + "/documents",
        files={"file": ("policy.md", b"Annual leave is 20 days.", "text/markdown")},
    )
    assert r.status_code == 200, r.text
    r = client.post(
        "/api/collections/" + cid + "/ask", json={"question": "How much annual leave?"}
    )
    assert r.status_code == 200, r.text
    assert len(r.json()["citations"]) == 1 and len(r.json()["trace"]) == 2
    assert len(client.get("/api/collections/" + cid + "/history").json()) == 1
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
