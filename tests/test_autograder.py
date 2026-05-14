"""Module 8 Integration autograder.

11 tests per Section H of the build packet. Runs against a real Weaviate
service in CI plus the cached flan-t5-base model.
"""

import ast
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import weaviate
from sentence_transformers import SentenceTransformer

import rag_service as rs_mod
from index_helpers import index_corpus_if_needed

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")
EVAL_PATH = os.path.join(DATA_DIR, "rag_eval.jsonl")
LEARNER_TESTS_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "test_rag.py")
README_PATH = os.path.join(os.path.dirname(__file__), "..", "README.md")


def _wait_for_weaviate(url: str, timeout: int = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if weaviate.Client(url).is_ready():
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"Weaviate not ready at {url} within {timeout}s")


@pytest.fixture(scope="session", autouse=True)
def setup_corpus():
    _wait_for_weaviate(WEAVIATE_URL)
    client = weaviate.Client(WEAVIATE_URL)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    index_corpus_if_needed(client, CORPUS_PATH, embedder)


@pytest.fixture(scope="session")
def sample_query() -> str:
    return "how do I rebase a feature branch"


@pytest.fixture(scope="session")
def sample_contexts(sample_query):
    return rs_mod.retrieve(sample_query, k=5)


# ---------------------------------------------------------------------------
# 1. structural
# ---------------------------------------------------------------------------

def test_rag_module_imports():
    for name in (
        "retrieve",
        "build_prompt",
        "generate",
        "rag_pipeline",
        "groundedness_score",
        "evaluate_rag",
        "STOPWORDS",
        "ABSTAIN_PHRASES",
    ):
        assert hasattr(rs_mod, name), f"rag_service must define {name!r}"


# ---------------------------------------------------------------------------
# 2. retrieve
# ---------------------------------------------------------------------------

def test_retrieve_returns_dicts_with_required_keys(sample_contexts):
    assert isinstance(sample_contexts, list)
    assert 0 < len(sample_contexts) <= 5
    for ctx in sample_contexts:
        for key in ("doc_id", "title", "answer_text"):
            assert key in ctx, f"context dict missing key {key!r}"


# ---------------------------------------------------------------------------
# 3. build_prompt — exact template
# ---------------------------------------------------------------------------

EXPECTED_TEMPLATE = (
    'Answer the question using only the context. If the context does not contain the answer, say "I don\'t know."\n'
    "\n"
    "Context:\n"
    "[1] {title_1}: {answer_text_1}\n"
    "[2] {title_2}: {answer_text_2}\n"
    "[3] {title_3}: {answer_text_3}\n"
    "[4] {title_4}: {answer_text_4}\n"
    "[5] {title_5}: {answer_text_5}\n"
    "\n"
    "Question: {query}\n"
    "Answer:"
)


def test_build_prompt_matches_required_template(sample_query, sample_contexts):
    """Byte-for-byte equality check against the canonical prompt template.

    The methodology specifies an exact template — any deviation in spacing,
    blank lines, or marker format changes the generator's behavior and the
    resulting metrics. Compare the rendered prompt against the expected
    string verbatim.
    """
    prompt = rs_mod.build_prompt(sample_query, sample_contexts)
    context_lines = []
    for i, ctx in enumerate(sample_contexts, start=1):
        truncated = " ".join(ctx["answer_text"].split()[:80])
        context_lines.append(f"[{i}] {ctx['title']}: {truncated}")
    expected = (
        'Answer the question using only the context. If the context does not contain the answer, say "I don\'t know."\n'
        "\n"
        "Context:\n"
        + "\n".join(context_lines) + "\n"
        "\n"
        f"Question: {sample_query}\n"
        "Answer:"
    )
    assert prompt == expected, (
        "build_prompt output does not match the canonical template byte-for-byte.\n"
        f"--- expected ---\n{expected!r}\n--- got ---\n{prompt!r}"
    )


def test_build_prompt_truncates_each_answer_text_to_80_tokens(sample_query, sample_contexts):
    """Each context's answer_text portion of the rendered prompt must be at most 80 tokens."""
    prompt = rs_mod.build_prompt(sample_query, sample_contexts)
    # Extract each "[i] {title}: {answer_text...}" line and check the portion
    # after the colon is at most 80 whitespace tokens.
    for i, ctx in enumerate(sample_contexts, start=1):
        marker = f"[{i}] "
        idx = prompt.find(marker)
        assert idx >= 0
        # End of this context line: next \n[ or \n\nQuestion:
        end = prompt.find("\n", idx)
        line = prompt[idx + len(marker):end]
        assert ":" in line, f"context line {i} missing 'title: answer' separator"
        after_colon = line.split(":", 1)[1].strip()
        token_count = len(after_colon.split())
        assert token_count <= 80, (
            f"context {i} answer_text rendered with {token_count} tokens; must be <= 80"
        )


# ---------------------------------------------------------------------------
# 4. generate
# ---------------------------------------------------------------------------

def test_generate_returns_str():
    out = rs_mod.generate("Translate English to German: Hello.")
    assert isinstance(out, str)
    assert len(out) > 0


# ---------------------------------------------------------------------------
# 5. rag_pipeline
# ---------------------------------------------------------------------------

def test_rag_pipeline_returns_required_keys(sample_query):
    out = rs_mod.rag_pipeline(sample_query, k=5)
    assert isinstance(out, dict)
    for key in ("query", "answer", "contexts", "prompt"):
        assert key in out, f"rag_pipeline output missing key {key!r}"
    assert out["query"] == sample_query


# ---------------------------------------------------------------------------
# 6. groundedness
# ---------------------------------------------------------------------------

def test_groundedness_zero_for_unrelated_answer(sample_contexts):
    score = rs_mod.groundedness_score("xyzzy plugh frobnicate", sample_contexts)
    assert score <= 0.1, f"groundedness on unrelated answer = {score}; should be ~ 0"


def test_groundedness_high_for_grounded_answer(sample_contexts):
    """Build an answer from context tokens; groundedness should be >= 0.7."""
    # Take 10 content words from the first context's answer_text; these will
    # all appear in the concatenated context, so |answer ∩ context| / |answer|
    # is ~ 1.0 (modulo stopword removal, which is symmetric).
    text = sample_contexts[0]["answer_text"]
    words = re.findall(r"[A-Za-z0-9_]+", text)
    answer = " ".join(words[:30])
    score = rs_mod.groundedness_score(answer, sample_contexts)
    assert score >= 0.7, f"grounded-answer groundedness = {score}; should be >= 0.7"


# ---------------------------------------------------------------------------
# 7. evaluate_rag
# ---------------------------------------------------------------------------

def test_evaluate_rag_keys():
    out = rs_mod.evaluate_rag(EVAL_PATH)
    headline_keys = (
        "answer_keyword_recall_main",
        "borderline_abstain_rate",
        "mean_groundedness_main",
        "mean_groundedness_borderline",
    )
    for key in headline_keys + ("per_question",):
        assert key in out, f"evaluate_rag output missing key {key!r}"
    for key in headline_keys:
        v = out[key]
        assert isinstance(v, (int, float)), (
            f"{key} must be numeric, got {type(v).__name__} ({v!r})"
        )
        assert 0.0 <= float(v) <= 1.0, f"{key}={v} out of [0, 1]"
    pq = out["per_question"]
    assert isinstance(pq, list)
    assert len(pq) == 30, f"per_question length = {len(pq)}; expected 30"
    diffs = {row.get("difficulty") for row in pq}
    for required_diff in ("single_fact", "single_doc_synthesis", "borderline"):
        assert required_diff in diffs, (
            f"per_question must include rows tagged {required_diff!r}; saw {diffs}"
        )


# ---------------------------------------------------------------------------
# 8. learner-written tests
# ---------------------------------------------------------------------------

def test_learner_tests_complete():
    """AST meta-check on tests/test_rag.py.

    - >= 3 test functions
    - Each test function has >= 1 ast.Assert node
    - No remaining `pass` body and no `pytest.fail("Not implemented...")`
    """
    src = open(LEARNER_TESTS_PATH).read()
    tree = ast.parse(src)

    test_funcs = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    assert len(test_funcs) >= 3, f"need >= 3 test functions in test_rag.py, found {len(test_funcs)}"

    for fn in test_funcs:
        asserts = [n for n in ast.walk(fn) if isinstance(n, ast.Assert)]
        assert asserts, f"test function {fn.name!r} has no `assert` statements"

        # No bare-pass body
        if len(fn.body) == 1 and isinstance(fn.body[0], ast.Pass):
            pytest.fail(f"test function {fn.name!r} has a `pass`-only body")

        # No pytest.fail("Not implemented...") placeholder
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "fail"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "pytest"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and "Not implemented" in node.args[0].value
            ):
                pytest.fail(
                    f"test function {fn.name!r} still contains the `pytest.fail(\"Not implemented...\")` stub"
                )


# ---------------------------------------------------------------------------
# 9. README substance
# ---------------------------------------------------------------------------

def test_readme_substance():
    assert os.path.exists(README_PATH)
    body = open(README_PATH).read()
    assert "TODO (replace this README before submitting)" not in body, (
        "README still contains the starter TODO placeholder — rewrite it per Task 9 "
        "(Overview, Setup, How to run it, Eval output, Known limitations)."
    )
    assert len(body) >= 400, f"README too short ({len(body)} chars; need >= 400)"
    lower = body.lower()
    for needle in ("rag", "groundedness"):
        assert needle in lower, f"README must mention {needle!r}"
    for needle in (
        "answer_keyword_recall_main",
        "borderline_abstain_rate",
        "mean_groundedness_main",
        "mean_groundedness_borderline",
    ):
        assert needle in body, f"README must reference the eval result key {needle!r}"
