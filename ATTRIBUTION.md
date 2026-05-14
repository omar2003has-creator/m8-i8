# Attribution

The Module 8 RAG corpus is a derivative work assembled from public Stack Exchange content and the BEIR CQADupStack benchmark. Use of the corpus requires the attributions below.

## License — top-level notice

> **Stack Exchange content is licensed under Creative Commons Attribution-ShareAlike (CC BY-SA), with the specific version determined by each post's contribution date.** Posts in this corpus inherit the CC BY-SA version that applied to the underlying Stack Exchange contribution when it was made. Downstream users must preserve attribution and comply with the applicable CC BY-SA version for each post they redistribute.

No single CC BY-SA version applies to the corpus as a whole. The corpus is best described as **mixed CC BY-SA source content**.

## Per-post license versioning

Stack Exchange's public licensing terms (https://stackoverflow.com/help/licensing) version user contributions by the UTC timestamp on the contribution:

| Contribution date (UTC) | License |
|---|---|
| Before 2011-04-08 | CC BY-SA 2.5 |
| 2011-04-08 through before 2018-05-02 | CC BY-SA 3.0 |
| On or after 2018-05-02 | CC BY-SA 4.0 |

Each row in `data/corpus.jsonl` carries a `license_version` field naming the applicable version for that specific post, derived from the post's `CreationDate` in the source Stack Exchange dump. Downstream redistribution should preserve the per-post `license_version` so the appropriate CC BY-SA terms can be applied to each document.

The corpus combines question, body, and answer text. Where a question and its answer have different contribution dates, the more recent of the two governs the combined document — preserving the share-alike obligations of the newer contribution while remaining backward-compatible with the older one.

## Stack Exchange (primary upstream)

The question and answer text in `data/corpus.jsonl` is taken from the public Stack Exchange data dump. Each document's `source_url` field links to the canonical Stack Exchange question page where the original authors are credited.

- **Source:** https://archive.org/details/stackexchange
- **Dump snapshot date:** 2024-04-06 (per the `Posts.xml` file timestamps in the downloaded 7z archives — the archive.org listing labels this as the April 2024 dump).
- **Sites used:**
  - `softwareengineering.stackexchange.com` — labeled `programmers` in BEIR CQADupStack. The Stack Exchange site was renamed from `programmers.stackexchange.com` to `softwareengineering.stackexchange.com` in 2014; the BEIR corpus uses the older name as its subset identifier and this corpus follows BEIR's naming.
  - `webmasters.stackexchange.com`
  - `android.stackexchange.com`
- **Licensing reference:** https://stackoverflow.com/help/licensing
- **Required attribution text** (per Stack Exchange's data-dump terms): include a hyperlink to the original question on the canonical Stack Exchange site whenever a corpus document is reproduced. The `source_url` field on every corpus row provides this link.

## BEIR CQADupStack (corpus filtering & paraphrastic eval source)

The selection of question identifiers (which questions appear in our corpus) and the duplicate-question relations used as paraphrastic eval pairs come from the BEIR CQADupStack benchmark.

- **Source:** https://huggingface.co/datasets/BeIR/cqadupstack
- **Qrels source:** https://huggingface.co/datasets/BeIR/cqadupstack-qrels
- **Benchmark paper:** Thakur, Nandan; Reimers, Nils; Rücklé, Andreas; Srivastava, Abhishek; Gurevych, Iryna. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS 2021 Datasets and Benchmarks Track. https://arxiv.org/abs/2104.08663
- **Underlying CQADupStack paper:** Hoogeveen, Doris; Verspoor, Karin M.; Baldwin, Timothy. *CQADupStack: A Benchmark Data Set for Community Question-Answering Research.* ADCS 2015.
- **License:** CC BY-SA 4.0.

## Required user-facing notice for downstream redistribution

Downstream uses of this corpus (classroom slides, RAG demos, internal documentation, derived datasets) must include a notice along the lines of:

> The corpus is derived from the public Stack Exchange data dump (https://archive.org/details/stackexchange) and the BEIR CQADupStack benchmark (https://github.com/beir-cellar/beir). User-contributed content is © its respective Stack Exchange authors and is licensed under Creative Commons Attribution-ShareAlike (CC BY-SA), with the version determined per post by contribution date (see Stack Exchange's licensing page at https://stackoverflow.com/help/licensing). Each post's specific CC BY-SA version is recorded in the corpus's `license_version` field; redistribution must preserve these per-post terms.
