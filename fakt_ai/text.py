"""Plain text handling shared by the stages that read sources."""

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

_STOPWORDS = frozenset(
    [
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "not",
        "but",
        "its",
        "their",
        "they",
        "than",
        "then",
        "there",
        "which",
        "who",
        "whom",
        "what",
        "when",
        "where",
        "why",
        "how",
        "can",
        "could",
        "may",
        "might",
        "will",
        "would",
        "should",
        "all",
        "any",
        "some",
        "more",
        "most",
        "such",
        "into",
        "over",
        "also",
        "been",
        "being",
        "about",
    ]
)
_WORD = re.compile(r"[a-z0-9][a-z0-9-]{2,}")
_FINDINGS_WORDS = ("conclusion", "we found", "results", "findings", "no significant")
_MAX_PARAGRAPH = 1500
_MIN_QUOTE_CHARS = 20
_GAP = "\n[...]\n"
_QUOTE_CONTEXT_CHARS = 900


def fence(label: str, text: str) -> str:
    """
    Wrap internet text in a labelled block it cannot close from the inside.

    Parameters
    ----------
    label : str
        Tag name for the block.
    text : str
        Untrusted text.

    Returns
    -------
    str
        The delimited text.
    """
    safe = re.sub(rf"<(/?)\s*{re.escape(label)}", "[", text, flags=re.IGNORECASE)
    return f"<{label}>\n{safe}\n</{label}>"


def normalise(text: str) -> str:
    """Fold typography, PDF line breaks and whitespace so quotes match across formats."""
    folded = unicodedata.normalize("NFKC", text)
    folded = folded.translate(
        str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
        | str.maketrans(dict.fromkeys("‐‑‒–—−", "-"))
    )
    folded = re.sub(r"(?<=[a-z])-\s*\n\s*(?=[a-z])", "", folded)
    return re.sub(r"\s+", " ", folded).strip()


def locate_quote(text: str, quote: str) -> str | None:
    """
    Find a quote in a text after normalising both.

    Parameters
    ----------
    text : str
        The document.
    quote : str
        The words said to come from it.

    Returns
    -------
    str | None
        The passage around the quote, or None when the quote is absent or too short to
        mean anything.
    """
    needle = _matchable(normalise(quote).strip(" .,;:'\""))
    if len(needle) < _MIN_QUOTE_CHARS:
        return None
    haystack = _matchable(normalise(text))
    start = haystack.find(needle)
    if start < 0:
        return None
    first = max(0, start - _QUOTE_CONTEXT_CHARS)
    return haystack[first : start + len(needle) + _QUOTE_CONTEXT_CHARS]


def pick_passages(text: str, query: str, budget: int) -> str:
    """
    Keep the paragraphs of a text that best match a query, in document order.

    Parameters
    ----------
    text : str
        The document.
    query : str
        What the passages should be about.
    budget : int
        Most characters to return.

    Returns
    -------
    str
        The whole text if it fits, otherwise its opening plus the best paragraphs.
    """
    if len(text) <= budget:
        return text
    paragraphs = _paragraphs(text)
    scores = _bm25(paragraphs, query)
    chosen, used = {0}, len(paragraphs[0])
    for index in sorted(range(len(paragraphs)), key=lambda i: scores[i], reverse=True):
        cost = len(paragraphs[index]) + len(_GAP)
        if scores[index] > 0 and index not in chosen and used + cost <= budget:
            chosen.add(index)
            used += cost
    return _GAP.join(paragraphs[i] for i in sorted(chosen))[:budget]


def overlapping_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Split text into pieces of `size` characters sharing `overlap` at each seam."""
    if len(text) <= size:
        return [text]
    step = size - overlap
    return [text[start : start + size] for start in range(0, len(text) - overlap, step)]


def _matchable(text: str) -> str:
    """Ignore case and hyphens, which line breaks in PDFs make unreliable."""
    return text.lower().replace("-", "")


def _paragraphs(text: str) -> list[str]:
    pieces = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        while len(block) > _MAX_PARAGRAPH:
            cut = block.rfind(". ", 0, _MAX_PARAGRAPH) + 1 or _MAX_PARAGRAPH
            pieces.append(block[:cut].strip())
            block = block[cut:].strip()
        if block:
            pieces.append(block)
    return pieces or [text[:_MAX_PARAGRAPH]]


def _bm25(paragraphs: Iterable[str], query: str) -> list[float]:
    documents = [Counter(_terms(p)) for p in paragraphs]
    lowered = [p.lower() for p in paragraphs]
    terms = set(_terms(query))
    count = len(documents)
    average = sum(sum(d.values()) for d in documents) / max(count, 1) or 1.0
    document_frequency = {t: sum(t in d for d in documents) for t in terms}
    scores = []
    for document, words in zip(documents, lowered):
        score = sum(
            math.log(1 + (count - document_frequency[t] + 0.5) / (document_frequency[t] + 0.5))
            * document[t]
            * 2.2
            / (document[t] + 1.2 * (0.25 + 0.75 * sum(document.values()) / average))
            for t in terms
            if t in document
        )
        scores.append(score * (1.25 if any(w in words for w in _FINDINGS_WORDS) else 1.0))
    return scores


def _terms(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS]
