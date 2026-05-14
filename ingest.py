"""Driver: stand up Weaviate, ingest the corpus once.

Usage (after Weaviate is running on localhost:8080):
    python ingest.py
"""

import os

import weaviate
from sentence_transformers import SentenceTransformer

from index_helpers import index_corpus_if_needed

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "data", "corpus.jsonl")


def main() -> None:
    client = weaviate.Client(WEAVIATE_URL)
    if not client.is_ready():
        raise SystemExit(f"Weaviate not reachable at {WEAVIATE_URL}")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    count = index_corpus_if_needed(client, CORPUS_PATH, embedder)
    print(f"Ingested/verified {count} objects in class 'Post'.")


if __name__ == "__main__":
    main()
