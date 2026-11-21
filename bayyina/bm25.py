import math
from collections import Counter

from .normalize import tokenize
from .types import Chunk


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._tf: dict[str, Counter] = {}
        self._dl: dict[str, int] = {}
        self._df: Counter = Counter()
        self._total = 0

    def add(self, chunks: list[Chunk]) -> None:
        for ch in chunks:
            toks = tokenize(ch.text)
            tf = Counter(toks)
            self._tf[ch.chunk_id] = tf
            self._dl[ch.chunk_id] = len(toks)
            self._total += len(toks)
            for term in tf:
                self._df[term] += 1

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        terms = set(tokenize(query))
        n = len(self._tf)
        if not terms or n == 0 or top_k <= 0:
            return []
        avgdl = self._total / n
        scores: list[tuple[str, float]] = []
        for cid, tf in self._tf.items():
            dl = self._dl[cid]
            norm = 1 - self.b + (self.b * dl / avgdl if avgdl else 0.0)
            score = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if not f:
                    continue
                df = self._df[t]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                score += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
            if score > 0:
                scores.append((cid, score))
        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores[:top_k]