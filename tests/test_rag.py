"""Learner-written tests for the RAG pipeline."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_service import retrieve, build_prompt, groundedness_score


def test_retrieve_returns_k_or_fewer():
    """retrieve(query, k=5) returns at most 5 dicts with required keys."""
    results = retrieve("how do I reverse a string in Python", k=5)
    assert isinstance(results, list)
    assert len(results) <= 5
    for item in results:
        assert "doc_id" in item
        assert "title" in item
        assert "answer_text" in item


def test_build_prompt_includes_context():
    """build_prompt includes instruction, numbered context lines, and Answer:"""
    query = "What is a REST API?"
    contexts = [
        {"doc_id": f"test:{i}", "title": f"Title {i}", "answer_text": f"Answer content {i} " * 10}
        for i in range(1, 6)
    ]
    prompt = build_prompt(query, contexts)

    # Instruction line
    assert 'say "I don\'t know."' in prompt

    # Numbered context lines
    for i in range(1, 6):
        assert f"[{i}] Title {i}:" in prompt

    # Question and Answer marker
    assert f"Question: {query}" in prompt
    assert prompt.strip().endswith("Answer:")


def test_groundedness_zero_for_unrelated_answer():
    """groundedness_score returns ~0.0 when answer shares no content words with context."""
    contexts = [
        {"answer_text": "the quick brown fox jumps over the lazy dog"},
        {"answer_text": "python is a programming language used for scripting"},
    ]
    score = groundedness_score("xyzzy plugh quux frobnicate", contexts)
    assert score <= 0.1