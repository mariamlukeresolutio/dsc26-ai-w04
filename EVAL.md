"""Compute recall@3 for BM25, dense and hybrid on the public queries.

Run from the repo root:  python scripts/eval_recall.py
Prints a markdown table to paste into EVAL.md.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bayyina.bm25 import BM25Index
from bayyina.chunking import chunk_text
from bayyina.embedder import HashingEmbedder
from bayyina.hybrid import HybridRetriever

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
K = 3


def read_exact(path: Path) -> str:
    # newline="" keeps \r\n as is, so offsets match the original file
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


docs = {p.stem: read_exact(p) for p in sorted((DATA / "contracts").glob("*.txt"))}
queries = [json.loads(line) for line in (DATA / "queries.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

chunks = [c for d, t in docs.items() for c in chunk_text(d, t, max_chars=500, overlap=50)]
by_id = {c.chunk_id: c for c in chunks}
emb = HashingEmbedder()
bm25 = BM25Index()
bm25.add(chunks)
hybrid = HybridRetriever(chunks, emb)
matrix = emb.embed([c.text for c in chunks])


def found(span_list, relevant):
    return any(
        s[0] == r["doc_id"] and s[1] < r["end"] and r["start"] < s[2]
        for s in span_list for r in relevant
    )


def top_bm25(q):
    return [by_id[i] for i, _ in bm25.search(q, top_k=K)]


def top_dense(q):
    scores = matrix @ emb.embed([q])[0]
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], chunks[i].chunk_id))
    return [chunks[i] for i in order[:K]]


def top_hybrid(q):
    return hybrid.search(q, top_k=K)


rows = []
for name, fn in [("BM25 only", top_bm25), ("Dense only (HashingEmbedder)", top_dense), ("Hybrid (RRF)", top_hybrid)]:
    missed = []
    for qd in queries:
        spans = [(h.doc_id, h.start, h.end) for h in fn(qd["query"])]
        if not found(spans, qd["relevant"]):
            missed.append(qd["qid"])
    rows.append((name, (len(queries) - len(missed)) / len(queries), missed))

print(f"{len(docs)} contracts, {len(chunks)} chunks, {len(queries)} queries\n")
print("| Retriever | recall@3 | queries missed |")
print("|---|---|---|")
for name, r, missed in rows:
    print(f"| {name} | {r:.2f} | {', '.join(missed) or 'none'} |")

Evaluation
Configuration:

chunk_text(..., max_chars=500, overlap=50)
HashingEmbedder()
Retriever	Recall@3
BM25	0.917
Dense	0.917
Hybrid	0.917
Evaluated on the 12 public queries in data/queries.jsonl.