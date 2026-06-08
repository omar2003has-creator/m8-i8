# Module 8 Integration — RAG Service

## Overview

This service implements a complete Retrieval-Augmented Generation (RAG) pipeline
over a Stack Exchange technical Q&A corpus. Given a user query, the pipeline
retrieves the top-5 relevant posts from Weaviate using hybrid search (BM25 + dense,
alpha=0.5), injects the retrieved answer content into a structured prompt, and
generates an answer using google/flan-t5-base with greedy decoding. The pipeline
is evaluated on 30 labeled questions using four metrics: answer_keyword_recall_main,
borderline_abstain_rate, mean_groundedness_main, and mean_groundedness_borderline.

## Setup

Start Weaviate, install dependencies, and ingest the corpus:

    docker run -d -p 8080:8080 semitechnologies/weaviate:1.24.1
    pip install -r requirements.txt
    python ingest.py

## How to Run

    from rag_service import rag_pipeline
    result = rag_pipeline("How do I handle 404 errors in Apache?", k=5)
    print(result["answer"])

Response keys: query, answer, contexts, prompt.

## Eval Output

Canonical run metrics (flan-t5-base, greedy decoding, alpha=0.5):

    answer_keyword_recall_main:   ~0.00-0.10
    borderline_abstain_rate:      ~0.80-1.00
    mean_groundedness_main:       ~0.30-0.60
    mean_groundedness_borderline: ~0.10-0.30

## Known Limitations

1. answer_keyword_recall_main is expected to be ~0.00-0.10. This is not a bug.
   flan-t5-base abstains on 21-23 of 25 answerable rows under the abstention prompt
   even when the context contains the answer. This is structural to the model family.

2. 512-token input ceiling. With 5 contexts of 80 tokens each plus overhead,
   the input is often truncated, cutting off context from later retrieved documents.

3. groundedness_score is blind to paraphrase. The metric measures exact content-word
   overlap, so correct paraphrases score low even when the answer is grounded.

4. STOPWORDS list is English-only. Code-heavy answers may produce unreliable
   groundedness scores since code tokens are not filtered.