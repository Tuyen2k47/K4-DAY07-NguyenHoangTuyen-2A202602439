import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import SentenceChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OLLAMA_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GeminiEmbedder,
    LocalEmbedder,
    OllamaEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

DEFAULT_DATA_DIR = Path("data/shopee-return-refund")


def load_documents_from_dir_or_files(data_dir: Path) -> list[Document]:
    """Load documents from directory with frontmatter parsing."""
    documents: list[Document] = []
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return documents

    for file_path in sorted(data_dir.glob("*.md")):
        raw_content = file_path.read_text(encoding="utf-8")
        metadata = {"source": str(file_path), "filename": file_path.name}
        content = raw_content

        if raw_content.startswith("---"):
            parts = raw_content.split("---", 2)
            if len(parts) >= 3:
                frontmatter_text = parts[1].strip()
                content = parts[2].strip()
                for line in frontmatter_text.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        metadata[k] = v

        doc_id = metadata.get("doc_id", file_path.stem)
        metadata["doc_id"] = doc_id
        documents.append(Document(id=doc_id, content=content, metadata=metadata))

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def make_ollama_llm(model_name: str = "qwen2.5:3b", host: str = "http://localhost:11434"):
    """Create an LLM function that queries local Ollama with real-time streaming."""
    def ollama_llm(prompt: str) -> str:
        import urllib.request
        import json
        url = f"{host.rstrip('/')}/api/generate"
        payload = json.dumps({"model": model_name, "prompt": prompt, "stream": True}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        collected = []
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                if line:
                    data = json.loads(line.decode("utf-8"))
                    chunk_text = data.get("response", "")
                    collected.append(chunk_text)
                    print(chunk_text, end="", flush=True)
                    if data.get("done", False):
                        break
        print()
        return "".join(collected)
    return ollama_llm


import json
import time

CACHE_FILE = Path("data/shopee_cache.json")


def load_cached_store(cache_path: Path, embedder) -> EmbeddingStore | None:
    """Load pre-computed chunks and embeddings from JSON cache."""
    if not cache_path.exists():
        return None
    try:
        t0 = time.time()
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        store = EmbeddingStore(collection_name="shopee_store", embedding_fn=embedder)
        store._store = data
        elapsed_ms = (time.time() - t0) * 1000
        print(f"\n[CACHE HIT] Loaded {len(data)} pre-embedded chunks from {cache_path} in {elapsed_ms:.1f}ms!")
        return store
    except Exception as err:
        print(f"\n[CACHE MISS] Failed to load cache ({err}), rebuilding...")
        return None


def save_store_cache(store: EmbeddingStore, cache_path: Path) -> None:
    """Save computed store records to JSON cache."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(store._store, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[CACHE SAVED] Saved {len(store._store)} chunks with vectors to {cache_path}")
    except Exception as err:
        print(f"Warning: Failed to save cache: {err}")


def run_shopee_demo(question: str | None = None, data_dir_path: Path | None = None) -> int:
    target_dir = data_dir_path or DEFAULT_DATA_DIR
    query = question or "Thời gian tối đa để gửi yêu cầu trả hàng hoàn tiền đối với đơn hàng Shopee là bao lâu?"

    print("=== Shopee Return & Refund RAG Demo ===")
    print(f"Data directory: {target_dir}")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "ollama").strip().lower()
    ollama_model = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:3b")

    llm_function = demo_llm

    if provider in ("ollama", "local"):
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
        try:
            llm_function = make_ollama_llm(model_name=ollama_model)
        except Exception as err:
            print(f"Failed to initialize Ollama LLM ({err}), using demo LLM")
            llm_function = demo_llm
    elif provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "gemini":
        try:
            embedder = GeminiEmbedder(model_name=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"Embedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    # Check for Vector Cache
    store = load_cached_store(CACHE_FILE, embedder)

    if store is None:
        docs = load_documents_from_dir_or_files(target_dir)
        if not docs:
            print("\nNo valid documents were loaded.")
            return 1

        print(f"\nLoaded {len(docs)} Shopee policy documents:")
        for doc in docs:
            print(f"  - [{doc.id}] {doc.metadata.get('title', doc.id)} (Audience: {doc.metadata.get('audience', 'N/A')})")

        max_sentences = 3
        chunker = SentenceChunker(max_sentences_per_chunk=max_sentences)
        chunked_docs: list[Document] = []
        for doc in docs:
            sub_chunks = chunker.chunk(doc.content)
            for idx, text_segment in enumerate(sub_chunks):
                chunked_docs.append(
                    Document(
                        id=f"{doc.id}_sentence_{idx}",
                        content=text_segment,
                        metadata={**doc.metadata, "doc_id": doc.id, "chunk_index": idx},
                    )
                )

        print(f"\nStrategy: SentenceChunker (max {max_sentences} sentences/chunk)")
        print(f"Total sentence chunks created: {len(chunked_docs)}")
        print("Computing embeddings for all chunks (first run only)...")
        store = EmbeddingStore(collection_name="shopee_store", embedding_fn=embedder)
        store.add_documents(chunked_docs)
        save_store_cache(store, CACHE_FILE)

    print(f"Active collection size: {store.get_collection_size()} sentence chunks")
    
    print("\n=== Vector Search Test (Unfiltered) ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        aud = result['metadata'].get('audience', 'N/A')
        print(f"{index}. score={result['score']:.3f} | doc_id={result['metadata'].get('doc_id')} | audience={aud}")
        print(f"   content: {result['content'][:140].replace(chr(10), ' ')}...")

    print("\n=== Vector Search Test (Filtered: audience='buyer') ===")
    filtered_results = store.search_with_filter(query, top_k=3, metadata_filter={"audience": "buyer"})
    for index, result in enumerate(filtered_results, start=1):
        aud = result['metadata'].get('audience', 'N/A')
        print(f"{index}. score={result['score']:.3f} | doc_id={result['metadata'].get('doc_id')} | audience={aud}")
        print(f"   content: {result['content'][:140].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test (Ollama LLM) ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_function)
    print(f"Question: {query}")
    print("\nAgent answer:")
    answer_result = agent.answer(query, top_k=3)
    if llm_function == demo_llm:
        print(answer_result)
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_shopee_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())
