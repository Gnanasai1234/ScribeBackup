"""Dependency-free ROUGE-1/2/L metric implementation."""
from __future__ import annotations

import re
from collections import Counter


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def _prf(overlap: int, generated_count: int, reference_count: int) -> dict[str, float]:
    precision = overlap / generated_count if generated_count else 0.0
    recall = overlap / reference_count if reference_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def _ngrams(words: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1)))


def rouge_n(generated: str, reference: str, n: int) -> dict[str, float]:
    pred, gold = _ngrams(tokens(generated), n), _ngrams(tokens(reference), n)
    overlap = sum((pred & gold).values())
    return _prf(overlap, sum(pred.values()), sum(gold.values()))


def rouge_l(generated: str, reference: str) -> dict[str, float]:
    pred, gold = tokens(generated), tokens(reference)
    # Rolling rows keep memory proportional to the reference length.
    previous = [0] * (len(gold) + 1)
    for token in pred:
        current = [0]
        for index, expected in enumerate(gold, start=1):
            current.append(previous[index - 1] + 1 if token == expected else max(previous[index], current[-1]))
        previous = current
    return _prf(previous[-1], len(pred), len(gold))


def calculate_rouge(generated: str, reference: str) -> dict[str, dict[str, float]]:
    return {"rouge1": rouge_n(generated, reference, 1), "rouge2": rouge_n(generated, reference, 2), "rougeL": rouge_l(generated, reference)}
