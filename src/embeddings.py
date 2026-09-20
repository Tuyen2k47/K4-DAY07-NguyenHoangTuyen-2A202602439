from __future__ import annotations

import hashlib
import math
import os
import re

# Multilingual model suitable for the Vietnamese corpora used in this Lab.
# The local backend remains optional; required checkpoints use MockEmbedder.
LOCAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
OLLAMA_EMBEDDING_MODEL = "llama3.1:latest"
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"


STOP_WORDS = {
    "shopee", "và", "hoặc", "của", "cho", "các", "có", "là", "được",
    "trong", "khi", "sau", "đến", "từ", "với", "đã", "sẽ", "đang",
    "về", "để", "thì", "mà", "bởi", "tại", "này", "đó", "những",
    "một", "theo", "nào", "bạn", "lại", "ra", "vào", "ở", "gì"
}


class MockEmbedder:
    """Deterministic embedding backend using n-gram feature hashing for keyword & semantic matching."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._backend_name = "mock embeddings fallback"

    def __call__(self, text: str) -> list[float]:
        clean_text = text.lower()
        words = re.findall(r"[\w]+", clean_text)
        if not words:
            return [0.0] * self.dim

        tokens_weighted: list[tuple[str, float]] = []
        for w in words:
            weight = 0.15 if w in STOP_WORDS else 1.0
            tokens_weighted.append((w, weight))

        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            if w1 not in STOP_WORDS or w2 not in STOP_WORDS:
                tokens_weighted.append((f"{w1}_{w2}", 2.0))

        for i in range(len(words) - 2):
            w1, w2, w3 = words[i], words[i+1], words[i+2]
            if any(w not in STOP_WORDS for w in (w1, w2, w3)):
                tokens_weighted.append((f"{w1}_{w2}_{w3}", 1.5))

        vector = [0.0] * self.dim
        for token, weight in tokens_weighted:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if ((h >> 16) & 1) else -1.0
            vector[idx] += sign * weight

        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector


class OllamaEmbedder:
    """Ollama API-backed local embedder with fallback."""

    def __init__(self, model_name: str = OLLAMA_EMBEDDING_MODEL, host: str = "http://localhost:11434") -> None:
        self.model_name = model_name
        self.host = host.rstrip("/")
        self._fallback = MockEmbedder()
        self._backend_name = f"ollama ({model_name})"

    def __call__(self, text: str) -> list[float]:
        import json
        import urllib.request

        prompt_text = text[:2000] if len(text) > 2000 else text
        url = f"{self.host}/api/embeddings"
        payload = json.dumps({"model": self.model_name, "prompt": prompt_text}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [float(v) for v in data["embedding"]]
        except Exception:
            return self._fallback(text)


class LocalEmbedder:
    """Sentence Transformers-backed local embedder."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._backend_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return [float(value) for value in embedding]


class OpenAIEmbedder:
    """OpenAI embeddings API-backed embedder."""

    def __init__(self, model_name: str = OPENAI_EMBEDDING_MODEL) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self._backend_name = model_name
        self.client = OpenAI()

    def __call__(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_name, input=text)
        return [float(value) for value in response.data[0].embedding]


class GeminiEmbedder:
    """Google Gemini embeddings API-backed embedder (google-genai SDK).

    Free-tier alternative to OpenAI for students without an OpenAI key —
    a Gemini API key (aistudio.google.com) has a free quota, no billing card needed.
    """

    def __init__(self, model_name: str = GEMINI_EMBEDDING_MODEL) -> None:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required for GeminiEmbedder")
        self.model_name = model_name
        self._backend_name = model_name
        self.client = genai.Client(api_key=api_key)

    def __call__(self, text: str) -> list[float]:
        response = self.client.models.embed_content(model=self.model_name, contents=text)
        return [float(value) for value in response.embeddings[0].values]


_mock_embed = MockEmbedder()
