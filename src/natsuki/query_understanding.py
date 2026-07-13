"""Query expansion via the Mistral API: adds related terms/synonyms to a
query before it hits BM25/dense retrieval, aiming to close vocabulary-
mismatch gaps (query says "brain scan", doc says "cerebral MRI")."""

from __future__ import annotations

from collections.abc import Callable

from natsuki.llm import chat

_SYSTEM_PROMPT = (
    "You expand search queries for an information-retrieval system. Given a "
    "query, output 3-5 related terms or synonyms that would help retrieve "
    "relevant documents -- one per line, lowercase, no numbering, no "
    "commentary, no repeating words already in the query."
)


def expand_query(query: str, chat_fn: Callable[[list[dict[str, str]]], str] = chat) -> str:
    """Returns the original query with LLM-generated related terms appended.
    Falls back to the original query unchanged if the expansion call fails
    or returns nothing usable -- retrieval should never break because an
    LLM call did."""
    try:
        raw = chat_fn(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ]
        )
    except Exception:
        return query

    terms = [line.strip() for line in raw.splitlines() if line.strip()]
    if not terms:
        return query
    return f"{query} {' '.join(terms)}"
