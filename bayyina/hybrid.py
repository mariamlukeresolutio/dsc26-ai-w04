from dataclasses import replace

from .bm25 import BM25Index
from .types import Chunk, Embedder, Hit, Reranker


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        rank = 0
        for cid in ranking:
            if cid in seen:
                continue
            seen.add(cid)
            rank += 1
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def _rows(vectors) -> list[list[float]]:
    if hasattr(vectors, "tolist"):
        vectors = vectors.tolist()
    return [[float(x) for x in row] for row in vectors]


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, *, k1: float = 1.5,
                 b: float = 0.75, reranker: Reranker | None = None):
        self._chunks = {c.chunk_id: c for c in chunks}
        self._embedder = embedder
        self._reranker = reranker
        self._bm25 = BM25Index(k1=k1, b=b)
        self._bm25.add(chunks)
        self._ids = list(self._chunks)
        self._vecs = _rows(embedder.embed([self._chunks[i].text for i in self._ids])) if self._ids else []

    def search(self, query: str, top_k: int = 5, candidates: int = 50) -> list[Hit]:
        if not query or not query.strip() or not self._ids:
            return []
        lexical = [cid for cid, _ in self._bm25.search(query, top_k=candidates)]
        q = _rows(self._embedder.embed([query]))[0]
        dots = [(cid, sum(a * b for a, b in zip(q, v))) for cid, v in zip(self._ids, self._vecs)]
        dots.sort(key=lambda x: (-x[1], x[0]))
        dense = [cid for cid, _ in dots[:candidates]]
        fused = rrf([lexical, dense])[:candidates]
        hits = []
        for cid, score in fused:
            c = self._chunks[cid]
            hits.append(Hit(chunk_id=c.chunk_id, doc_id=c.doc_id, start=c.start, end=c.end,
                            text=c.text, score=score))
        if self._reranker is not None:
            new = self._reranker(query, hits)
            hits = [replace(h, score=float(s)) for h, s in zip(hits, new)]
            hits.sort(key=lambda h: (-h.score, h.chunk_id))
        return hits[:top_k]