"""Ollama HTTP client, injected at the application boundary for testing."""

import os
import math
import httpx


class ModelError(Exception):
    pass


class Ollama:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        self.model = os.getenv("CHAT_MODEL", "qwen3:8b")
        self.embedding_model = os.getenv("EMBED_MODEL", "nomic-embed-text")

    def _post(self, path, payload):
        try:
            with httpx.Client(timeout=180) as client:
                response = client.post(self.base_url + path, json=payload)
                response.raise_for_status()
                result = response.json()
            if not isinstance(result, dict):
                raise ValueError("Expected response object")
            return result
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelError(
                "Could not get a valid model response. Check Ollama and the configured model."
            ) from exc

    def chat(self, messages):
        result = self._post(
            "/api/chat",
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
        )
        message = result.get("message")
        if (
            not isinstance(message, dict)
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise ModelError("The model returned an empty or invalid message.")
        return message["content"]

    def embed(self, texts):
        result = self._post(
            "/api/embed",
            {"model": self.embedding_model, "input": texts, "truncate": False},
        )
        vectors = result.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ModelError("Embedding response count does not match the input.")
        dimension = None
        for v in vectors:
            if (
                not isinstance(v, list)
                or not v
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in v)
            ):
                raise ModelError("Invalid embedding vector.")
            if not any(x != 0 for x in v):
                raise ModelError("Zero embedding vector.")
            if dimension is not None and len(v) != dimension:
                raise ModelError("Inconsistent embedding dimensions.")
            dimension = len(v)
        return vectors

    def tool_chat(self, messages, tools):
        result = self._post(
            "/api/chat",
            {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "stream": False,
                "options": {"temperature": 0, "num_ctx": 16384},
            },
        )
        message = result.get("message")
        if not isinstance(message, dict) or not isinstance(
            message.get("content", ""), str
        ):
            raise ModelError("The model returned an invalid agent message.")
        if message.get("tool_calls") is not None and not isinstance(
            message["tool_calls"], list
        ):
            raise ModelError("The model returned invalid tool calls.")
        return {**message, "role": "assistant"}
