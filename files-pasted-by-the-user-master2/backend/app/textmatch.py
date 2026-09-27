"""Whole-word lexical matching helpers.

Plain substring containment (`term in text`) produces false positives —
e.g. the keyword "port" matches inside "important", "reports", "support" —
which corrupts keyword-overlap ranking used for sentence/chunk selection
and for the Ask Document lexical retrieval fallback. These helpers anchor
on word boundaries instead.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable


@lru_cache(maxsize=2048)
def _pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(term)}\b")


def term_hits(haystack: str, terms: Iterable[str]) -> int:
    """Count how many of `terms` occur as whole words in `haystack` (case-insensitive)."""
    haystack_l = haystack.lower()
    return sum(1 for t in terms if _pattern(t).search(haystack_l))


def term_present(haystack: str, terms: Iterable[str]) -> bool:
    """True if any of `terms` occurs as a whole word in `haystack` (case-insensitive)."""
    haystack_l = haystack.lower()
    return any(_pattern(t).search(haystack_l) for t in terms)
