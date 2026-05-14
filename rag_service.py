"""Module 8 — Integration Task: RAG Service.

Implement a complete RAG mini-service: retrieve top-k context from Weaviate,
construct a context-injected prompt, call flan-t5-base, return the answer with
the retrieved context, and evaluate end-to-end on a 30-question eval set.

Methodology (canonical — autograder enforces; do not deviate):
- Top-k: k=5. Hybrid retrieval with alpha=0.5.
- Prompt template (exact, byte-for-byte):

      Answer the question using only the context. If the context does not contain the answer, say "I don't know."

      Context:
      [1] {title_1}: {answer_text_1}
      [2] {title_2}: {answer_text_2}
      [3] {title_3}: {answer_text_3}
      [4] {title_4}: {answer_text_4}
      [5] {title_5}: {answer_text_5}

      Question: {query}
      Answer:

  Each {answer_text_i} is the retrieved post's `answer_text` field truncated
  to its first 80 whitespace-split tokens.

- Generator: google/flan-t5-base, greedy decoding (num_beams=1, max_new_tokens=128),
  loaded ONCE at module level (not per call).

- Row-class definitions (in data/rag_eval.jsonl):
    answerable: difficulty in {single_fact, single_doc_synthesis} — 25 rows
    borderline: difficulty == "borderline" — 5 rows (supporting doc does NOT
                contain the answer; correct behavior is to abstain)

- answer_keyword_recall_main: over 25 answerable rows. For each row,
    (# expected_answer_keywords matched in answer, case-insensitive WHOLE-WORD)
    / (# total expected_answer_keywords).
  Average across rows. **Expected canonical baseline ~ 0.00–0.10.**

- borderline_abstain_rate: over 5 borderline rows. Correctly abstained iff
    (a) lowercased answer contains any phrase in ABSTAIN_PHRASES, OR
    (b) len(answer.strip()) <= 20 chars AND groundedness_score(answer, contexts) <= 0.2.

- mean_groundedness_main / mean_groundedness_borderline: groundedness_score
  averaged over the respective row classes.

- groundedness_score: lowercase both sides; tokenize on whitespace + punctuation;
  remove stopwords (the STOPWORDS constant below); compute
    |answer_tokens ∩ context_tokens| / |answer_tokens|.
  Empty answer -> 0.0. Score in [0, 1].
"""

import json
import os
import re
import string
from typing import List

import weaviate
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from index_helpers import bm25_search, dense_search, hybrid_search  # noqa: F401

# ---------------------------------------------------------------------------
# Constants — DO NOT MODIFY (autograder depends on these values)
# ---------------------------------------------------------------------------

CLASS_NAME = "Post"
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
GENERATOR_MODEL = "google/flan-t5-base"
EMBEDDER_MODEL = "all-MiniLM-L6-v2"

ABSTAIN_PHRASES = [
    "i don't know",
    "i do not know",
    "not in the context",
    "the context does not",
    "cannot be answered",
    "no information",
]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "he", "her", "him", "his", "i", "if", "in", "into", "is",
    "it", "its", "just", "may", "me", "might", "my", "no", "nor", "not",
    "of", "on", "or", "our", "out", "over", "she", "so", "some", "such",
    "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "to", "too", "under", "until", "up", "was",
    "we", "were", "what", "when", "where", "which", "while", "who",
    "will", "with", "would", "you", "your",
}

# ---------------------------------------------------------------------------
# Module-level model loading (loaded ONCE per process; do not reload per call)
# ---------------------------------------------------------------------------

_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL)
_embedder = SentenceTransformer(EMBEDDER_MODEL)

# Weaviate client is created lazily so importing this module does not crash
# when the Weaviate container is not yet running (mirrors the drill's
# weaviate_ready contract — connection failures surface at call time, not
# import time).
_client: weaviate.Client | None = None


def _get_client() -> weaviate.Client:
    global _client
    if _client is None:
        _client = weaviate.Client(WEAVIATE_URL)
    return _client


# ---------------------------------------------------------------------------
# Helpers (provided)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize_for_groundedness(text: str) -> set[str]:
    """Lowercase + whitespace+punctuation tokenize + stopword removal."""
    tokens = _TOKEN_RE.findall(text.lower())
    return {t for t in tokens if t and t not in STOPWORDS}


def _whole_word_match(keyword: str, answer: str) -> bool:
    """Case-insensitive whole-word match used by answer_keyword_recall_main."""
    pattern = r"\b" + re.escape(keyword) + r"\b"
    return re.search(pattern, answer, flags=re.IGNORECASE) is not None


# ---------------------------------------------------------------------------
# Functions to implement
# ---------------------------------------------------------------------------

def retrieve(query: str, k: int = 5) -> List[dict]:
    """Retrieve top-k contexts using hybrid_search with alpha=0.5.

    Return a list of dicts with at least: doc_id, title, answer_text.

    Use hybrid_search (provided in index_helpers.py) to get the top-k doc_ids,
    then resolve each doc_id back to its full record (title + answer_text)
    using the Weaviate client. The generator needs the answer content; the
    question portion is not used in the prompt context.
    """
    # TODO: call hybrid_search(_get_client(), query, k, _embedder, alpha=0.5) -> list[doc_id]
    # TODO: resolve each doc_id to {"doc_id", "title", "answer_text"}
    raise NotImplementedError("retrieve is not yet implemented")


def build_prompt(query: str, contexts: List[dict]) -> str:
    """Build the canonical prompt (see module docstring for the EXACT template).

    Each context's answer_text MUST be truncated to its first 80 whitespace-
    split tokens before insertion. Title and "[i]:" markers are kept.

    The autograder checks the template byte-for-byte; deviations fail
    test_build_prompt_matches_required_template.
    """
    # TODO: truncate each contexts[i]["answer_text"] to 80 whitespace tokens
    # TODO: assemble the exact template (see module docstring)
    raise NotImplementedError("build_prompt is not yet implemented")


def generate(prompt: str) -> str:
    """Call flan-t5-base with greedy decoding.

    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = _model.generate(**inputs, max_new_tokens=128, num_beams=1)
    return _tokenizer.decode(outputs[0], skip_special_tokens=True)
    """
    # TODO: tokenize, generate (greedy: num_beams=1, max_new_tokens=128), decode
    raise NotImplementedError("generate is not yet implemented")


def rag_pipeline(query: str, k: int = 5) -> dict:
    """Compose retrieve -> build_prompt -> generate.

    Return: {"query", "answer", "contexts", "prompt"}.
    """
    # TODO: contexts = retrieve(...); prompt = build_prompt(...); answer = generate(...)
    # TODO: return the four-key dict
    raise NotImplementedError("rag_pipeline is not yet implemented")


def groundedness_score(answer: str, contexts: List[dict]) -> float:
    """Content-word overlap between answer and concatenated contexts[*].answer_text.

    - Lowercase both sides.
    - Tokenize on whitespace + punctuation (use _tokenize_for_groundedness).
    - Remove stopwords (done by _tokenize_for_groundedness).
    - |answer ∩ context| / |answer|.
    - Empty answer -> 0.0.
    """
    # TODO: handle empty-answer edge case
    # TODO: tokenize answer and concatenated contexts using _tokenize_for_groundedness
    # TODO: return |answer ∩ context| / |answer|
    raise NotImplementedError("groundedness_score is not yet implemented")


def evaluate_rag(eval_path: str) -> dict:
    """Run the pipeline over the 30-pair eval set and produce the five-key dict.

    Each eval row in `data/rag_eval.jsonl` is a dict with keys:
      - "question": str (the user query — not "query")
      - "expected_answer_keywords": list[str]
      - "supporting_doc_id": str
      - "difficulty": one of "single_fact" | "single_doc_synthesis" | "borderline"

    Iterate eval pairs; partition rows by `difficulty`:
      answerable (single_fact / single_doc_synthesis) -> 25 rows
      borderline (difficulty == "borderline") -> 5 rows

    Compute:
      answer_keyword_recall_main   (answerable only; whole-word case-insensitive)
      mean_groundedness_main       (answerable only)
      borderline_abstain_rate      (borderline only; ABSTAIN_PHRASES OR short+low-grd)
      mean_groundedness_borderline (borderline only)
      per_question                 (one diagnostic dict per row)

    Return all five keys. The metric split is enforced by the autograder
    (test_evaluate_rag_keys); folding all 30 rows into one recall metric
    penalises correct abstention and is wrong here by design.
    """
    # TODO: load eval rows; partition by difficulty
    # TODO: run rag_pipeline on each row; compute per-row metrics
    # TODO: assemble the 5-key dict (4 aggregates + per_question list)
    raise NotImplementedError("evaluate_rag is not yet implemented")
