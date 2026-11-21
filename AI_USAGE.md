# AI usage

## Tools I used, and for what

I used Claude to write first drafts of `normalize.py`, `chunking.py`, `bm25.py`, `hybrid.py` and `cite.py` from the spec. I also used it to write a randomized checker for the chunking rules, a hand-written reference BM25 to compare scores against, and `scripts/eval_recall.py` for the recall table. I read the code, ran the public tests myself, and pasted the provided files (`types.py`, `embedder.py`, `pyproject.toml`, tests, queries) so the code matched the real interfaces.

## Where AI-written chunking or BM25 code broke one of the spec's rules

The first chunker, after a mid-word hard cut, restarted the next chunk at `end - overlap`. That offset can fall inside a run of whitespace, so the chunk began with whitespace. That breaks rule 1 (a chunk's text must not start with whitespace). The public tests did not catch it. A randomized property check did: it ran 3,000 random texts with random `max_chars` and `overlap` and checked rules 1 to 7 on each. The fix moves the start forward to the next non-whitespace character. The same session also caught a mismatch with the provided `types.py` (`chunk_id` instead of `id`, and a reranker that is a plain function), which I fixed once I pasted the real file.

## My recall@3 table (from EVAL.md) and why hybrid did or did not win

| Retriever | recall@3 | queries missed |
|---|---|---|
| BM25 only | TBD | TBD |
| Dense only (HashingEmbedder) | TBD | TBD |
| Hybrid (RRF) | TBD | TBD |

TBD: one sentence on why hybrid did or did not win, written after running `python scripts/eval_recall.py`.