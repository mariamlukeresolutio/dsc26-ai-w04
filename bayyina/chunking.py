from .types import Chunk

_SENT_END = ".!?\u061F\u06D4\n"


def _cut(window: str, max_chars: int) -> int:
    """Return c (length of the chunk before trailing-whitespace removal)."""
    lo = max(max_chars // 2, 1)
    for c in range(max_chars, lo - 1, -1):
        if c >= 2 and window[c - 2:c] == "\n\n":
            return c
    for c in range(max_chars, lo - 1, -1):
        if window[c - 1] in _SENT_END:
            return c
    for c in range(max_chars, lo - 1, -1):
        if window[c - 1].isspace():
            return c
    return max_chars


def chunk_text(doc_id: str, text: str, max_chars: int = 500, overlap: int = 50) -> list[Chunk]:
    if not (max_chars > 0 and 0 <= overlap < max_chars):
        raise ValueError("need max_chars > 0 and 0 <= overlap < max_chars")
    n = len(text)
    s = 0
    while s < n and text[s].isspace():
        s += 1
    chunks: list[Chunk] = []
    while s < n:
        if n - s <= max_chars:
            end = s + len(text[s:].rstrip())
            chunks.append(Chunk(doc_id=doc_id, chunk_id=f"{doc_id}#{len(chunks)}", start=s, end=end, text=text[s:end]))
            break
        c = _cut(text[s:s + max_chars], max_chars)
        end = s + len(text[s:s + c].rstrip())
        chunks.append(Chunk(doc_id=doc_id, chunk_id=f"{doc_id}#{len(chunks)}", start=s, end=end, text=text[s:end]))
        lo = max(end - overlap, s + 1)
        if text[end].isspace():
            # not a mid-word cut: next chunk must start at a word start
            ns = None
            for i in range(lo, end):
                if text[i - 1].isspace() and not text[i].isspace():
                    ns = i
                    break
            if ns is None:
                ns = end
                while ns < n and text[ns].isspace():
                    ns += 1
            s = ns
        else:
            # mid-word cut: any start is allowed, but never on whitespace
            while text[lo].isspace():
                lo += 1
            s = lo
    return chunks