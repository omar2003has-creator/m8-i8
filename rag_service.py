"""Module 8 — Integration Task: RAG Service."""

import json
import os
import re
from typing import List

import weaviate
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from index_helpers import hybrid_search  # noqa: F401


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
# Module-level model loading — loaded ONCE on import
# ---------------------------------------------------------------------------

_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL)
_embedder = SentenceTransformer(EMBEDDER_MODEL)

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


def _tokenize_for_groundedness(text: str) -> set:
    tokens = _TOKEN_RE.findall(text.lower())
    return {t for t in tokens if t and t not in STOPWORDS}


def _whole_word_match(keyword: str, answer: str) -> bool:
    pattern = r"\b" + re.escape(keyword) + r"\b"
    return re.search(pattern, answer, flags=re.IGNORECASE) is not None


# ---------------------------------------------------------------------------
# Task 2: retrieve
# ---------------------------------------------------------------------------

def retrieve(query: str, k: int = 5) -> List[dict]:
    """Retrieve top-k contexts using hybrid_search with alpha=0.5.
    Returns list of dicts with keys: doc_id, title, answer_text.
    """
    client = _get_client()
    qv = _embedder.encode(query, convert_to_numpy=True).tolist()

    result = (
        client.query
        .get(CLASS_NAME, ["doc_id", "title", "answer_text"])
        .with_hybrid(query=query, vector=qv, alpha=0.5)
        .with_limit(k)
        .do()
    )

    items = result.get("data", {}).get("Get", {}).get(CLASS_NAME, []) or []
    return [
        {
            "doc_id":      item["doc_id"],
            "title":       item["title"],
            "answer_text": item["answer_text"],
        }
        for item in items
    ]


# ---------------------------------------------------------------------------
# Task 3: build_prompt
# ---------------------------------------------------------------------------

def build_prompt(query: str, contexts: List[dict]) -> str:
    """Build the canonical prompt. Truncates each answer_text to 80 tokens."""
    context_lines = []
    for i, ctx in enumerate(contexts, start=1):
        truncated = " ".join(ctx["answer_text"].split()[:80])
        context_lines.append(f"[{i}] {ctx['title']}: {truncated}")

    context_block = "\n".join(context_lines)

    prompt = (
        'Answer the question using only the context. If the context does not contain the answer, say "I don\'t know."\n'
        "\n"
        "Context:\n"
        f"{context_block}\n"
        "\n"
        f"Question: {query}\n"
        "Answer:"
    )
    return prompt


# ---------------------------------------------------------------------------
# Task 4: generate
# ---------------------------------------------------------------------------

def generate(prompt: str) -> str:
    """Run the prompt through flan-t5-base with greedy decoding."""
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = _model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=128,
        num_beams=1,
    )
    return _tokenizer.decode(outputs[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Task 5: rag_pipeline
# ---------------------------------------------------------------------------

def rag_pipeline(query: str, k: int = 5) -> dict:
    """Full RAG: retrieve → build_prompt → generate → return."""
    contexts = retrieve(query, k)
    prompt = build_prompt(query, contexts)
    answer = generate(prompt)
    return {
        "query":    query,
        "answer":   answer,
        "contexts": contexts,
        "prompt":   prompt,
    }


# ---------------------------------------------------------------------------
# Task 6: groundedness_score
# ---------------------------------------------------------------------------

def groundedness_score(answer: str, contexts: List[dict]) -> float:
    """Content-word overlap between answer and concatenated contexts."""
    answer_tokens = _tokenize_for_groundedness(answer)
    if not answer_tokens:
        return 0.0

    context_text = " ".join(ctx["answer_text"] for ctx in contexts)
    context_tokens = _tokenize_for_groundedness(context_text)

    overlap = answer_tokens & context_tokens
    return len(overlap) / len(answer_tokens)


# ---------------------------------------------------------------------------
# Task 7: evaluate_rag
# ---------------------------------------------------------------------------

def evaluate_rag(eval_path: str) -> dict:
    """Run the pipeline over the 30-pair eval set."""
    rows = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    answerable_difficulties = {"single_fact", "single_doc_synthesis"}

    keyword_recalls = []
    groundedness_main = []
    groundedness_borderline = []
    abstain_scores = []
    per_question = []

    for idx, row in enumerate(rows):
        question   = row["question"]
        difficulty = row["difficulty"]
        keywords   = row.get("expected_answer_keywords", [])

        result   = rag_pipeline(question, k=5)
        answer   = result["answer"]
        contexts = result["contexts"]

        grd = groundedness_score(answer, contexts)
        answer_lower = answer.lower().strip()

        if difficulty in answerable_difficulties:
            # answer_keyword_recall
            if keywords:
                matched = [kw for kw in keywords if _whole_word_match(kw, answer)]
                recall = len(matched) / len(keywords)
            else:
                matched = []
                recall = 0.0

            keyword_recalls.append(recall)
            groundedness_main.append(grd)

            per_question.append({
                "row_index":        idx,
                "difficulty":       difficulty,
                "question":         question,
                "answer":           answer,
                "groundedness":     grd,
                "matched_keywords": matched,
            })

        elif difficulty == "borderline":
            # Check abstention
            phrase_abstained = any(phrase in answer_lower for phrase in ABSTAIN_PHRASES)
            short_and_ungrounded = (len(answer.strip()) <= 20 and grd <= 0.2)
            abstained = phrase_abstained or short_and_ungrounded

            abstain_scores.append(1 if abstained else 0)
            groundedness_borderline.append(grd)

            per_question.append({
                "row_index":    idx,
                "difficulty":   difficulty,
                "question":     question,
                "answer":       answer,
                "groundedness": grd,
                "abstained":    abstained,
            })

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "answer_keyword_recall_main":   mean(keyword_recalls),
        "borderline_abstain_rate":      mean(abstain_scores),
        "mean_groundedness_main":       mean(groundedness_main),
        "mean_groundedness_borderline": mean(groundedness_borderline),
        "per_question":                 per_question,
    }