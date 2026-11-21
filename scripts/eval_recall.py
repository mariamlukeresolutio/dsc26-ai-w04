import json
from pathlib import Path

from bayyina.bm25 import BM25Index
from bayyina.chunking import chunk_text
from bayyina.embedder import HashingEmbedder
from bayyina.hybrid import HybridRetriever

root = Path(".")

all_chunks = []

for path in sorted((root / "data" / "contracts").glob("*.txt")):
    doc_id = path.stem
    text = path.read_text(encoding="utf-8")

    doc_chunks = chunk_text(
        doc_id,
        text,
        max_chars=500,
        overlap=50,
    )

    all_chunks.extend(doc_chunks)

chunk_map = {c.chunk_id: c for c in all_chunks}

embedder = HashingEmbedder()

bm25 = BM25Index()
bm25.add(all_chunks)

hybrid = HybridRetriever(all_chunks, embedder)

# Dense retrieval setup
chunk_ids = [c.chunk_id for c in all_chunks]
chunk_vecs = embedder.embed([c.text for c in all_chunks])


def overlaps(chunk, start, end):
    return max(chunk.start, start) < min(chunk.end, end)


queries = []

with open(root / "data" / "queries.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            queries.append(json.loads(line))

bm25_hits = 0
dense_hits = 0
hybrid_hits = 0

for q in queries:
    query_text = q["query"]
    relevant = q["relevant"]

    # --------------------
    # BM25
    # --------------------
    bm25_top3 = [
        chunk_map[cid]
        for cid, _ in bm25.search(query_text, top_k=3)
    ]

    found = False
    for hit in bm25_top3:
        for rel in relevant:
            if (
                hit.doc_id == rel["doc_id"]
                and overlaps(hit, rel["start"], rel["end"])
            ):
                found = True
                break
        if found:
            break

    if found:
        bm25_hits += 1

    # --------------------
    # Dense
    # --------------------
    q_vec = embedder.embed([query_text])[0]

    dense_scores = []

    for cid, vec in zip(chunk_ids, chunk_vecs):
        score = sum(a * b for a, b in zip(q_vec, vec))
        dense_scores.append((cid, score))

    dense_scores.sort(key=lambda x: (-x[1], x[0]))

    dense_top3 = [
        chunk_map[cid]
        for cid, _ in dense_scores[:3]
    ]

    found = False
    for hit in dense_top3:
        for rel in relevant:
            if (
                hit.doc_id == rel["doc_id"]
                and overlaps(hit, rel["start"], rel["end"])
            ):
                found = True
                break
        if found:
            break

    if found:
        dense_hits += 1

    # --------------------
    # Hybrid
    # --------------------
    hybrid_top3 = hybrid.search(
        query_text,
        top_k=3,
        candidates=50,
    )

    found = False
    for hit in hybrid_top3:
        for rel in relevant:
            if (
                hit.doc_id == rel["doc_id"]
                and overlaps(hit, rel["start"], rel["end"])
            ):
                found = True
                break
        if found:
            break

    if found:
        hybrid_hits += 1

n = len(queries)

print(f"Queries evaluated: {n}")
print(f"BM25 Recall@3:   {bm25_hits}/{n} ({bm25_hits/n:.3f})")
print(f"Dense Recall@3:  {dense_hits}/{n} ({dense_hits/n:.3f})")
print(f"Hybrid Recall@3: {hybrid_hits}/{n} ({hybrid_hits/n:.3f})")