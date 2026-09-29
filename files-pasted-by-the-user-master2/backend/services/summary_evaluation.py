"""Auditable, deterministic quality metrics for generated document summaries.

The evaluator does not call a generative model.  Every score is derived from the
summary, source text, stored claims, or cited source chunks supplied by Scribe.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any

from services.local_fallbacks import basic_local_claim_check

EVALUATION_VERSION = "deterministic-rubric-v1"
WEIGHTS = {
    "relevance": 15,
    "completeness": 20,
    "factual_consistency": 20,
    "groundedness": 20,
    "summary_quality": 15,
    "validation_accuracy": 10,
}
METRIC_NAMES = tuple(WEIGHTS)
STOP_WORDS = set("a an and are as at be been but by can could did do does for from had has have he her hers him his how i in into is it its may might more most must my no not of on or our ours she should so than that the their them then there these they this those to up us was we were what when where which who why will with would you your".split())
IMPORTANT_CUES = re.compile(r"\b(increased?|decreased?|declined?|grew|risk|risks|recommend(?:s|ed|ation)?|decision|approved|rejected|must|will|committed|target|goal|concluded|conclusion|finding|result|revenue|cost|profit|loss|deadline|percent|\d+(?:[.,]\d+)?%?)\b", re.I)


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in re.findall(r"[^\W_]+(?:['’][^\W_]+)?", text, flags=re.UNICODE)
            if len(token) > 1 and token.casefold() not in STOP_WORDS]


def _sentences(text: str) -> list[str]:
    return [part.strip(" \t\r\n-*•") for part in re.split(r"(?<=[.!?。！？])\s+|\n+", text) if len(_tokens(part)) >= 3]


def _metric(score: float | None, explanation: str, *, status: str = "completed", evidence: list[Any] | None = None) -> dict[str, Any]:
    return {"score": round(max(0.0, min(100.0, score)), 2) if score is not None else None,
            "explanation": explanation, "status": status, "evidence": evidence or []}


def _claim_refs(claim: dict[str, Any], chunks_by_id: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    refs: list[dict[str, Any]] = []
    evidence: list[str] = []
    for source in claim.get("sources", []) or []:
        if not isinstance(source, dict):
            continue
        chunk_id = source.get("chunk_id")
        chunk = chunks_by_id.get(str(chunk_id)) if chunk_id else None
        if not chunk:
            continue
        source_page = source.get("page")
        page = source_page if isinstance(source_page, int) else chunk.get("page")
        section = source.get("section") if source.get("section") not in {None, "Document text"} else chunk.get("section")
        ref = {"document_id": source.get("document_id") or chunk.get("documentId"), "chunk_id": chunk_id}
        if page is not None:
            ref["page"] = page
        if section:
            ref["section"] = section
        refs.append(ref)
        quote = claim.get("evidence")
        evidence.append(str(quote).strip() if quote and str(quote).strip() else str(chunk.get("text") or ""))
    return refs, [item for item in evidence if item]


def _independent_source_evidence(claim: str, chunks: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    """Select up to three source chunks with the strongest lexical claim overlap."""
    claim_terms = set(_tokens(claim))
    if not claim_terms:
        return []
    scored = []
    for chunk in chunks:
        text = str(chunk.get("text") or "")
        terms = set(_tokens(text))
        overlap = len(claim_terms & terms) / len(claim_terms)
        if overlap:
            scored.append((overlap, chunk, text))
    return [(chunk, text) for _, chunk, text in sorted(scored, key=lambda row: row[0], reverse=True)[:3]]


def evaluate_summary(source_text: str, summary: str | None, *, claims: list[dict[str, Any]] | None = None,
                     chunks: list[dict[str, Any]] | None = None, source_language: str = "und",
                     summary_language: str | None = None, purpose: str = "", depth: str = "standard",
                     provider: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute six independent metrics and a normalized weighted overall score."""
    summary = (summary or "").strip()
    fingerprint = hashlib.sha256((summary + "\0" + str(source_language)).encode("utf-8")).hexdigest() if summary else None
    if not summary:
        return {
            "evaluation_status": "unavailable", "reason": "No usable summary was generated, so no quality scores were calculated.",
            "metrics": {name: _metric(None, "A usable summary is required for evaluation.", status="unavailable") for name in METRIC_NAMES},
            "overall_score": None, "overall_score_rounded": None, "applicable_metric_count": 0,
            "claims": [], "unsupported_claims": [], "warnings": ["Summary generation failed; evaluation was not run."],
            "evaluator": {"provider": "programmatic", "model": EVALUATION_VERSION}, "evaluation_version": EVALUATION_VERSION,
            "summary_fingerprint": fingerprint,
        }

    source_text = source_text or ""
    source_tokens, summary_tokens = _tokens(source_text), _tokens(summary)
    src_set = set(source_tokens)
    sum_counts = Counter(summary_tokens)
    sum_sents, src_sents = _sentences(summary), _sentences(source_text)
    claims = [c for c in (claims or []) if isinstance(c, dict) and str(c.get("claim") or "").strip()]
    chunks_by_id = {str(c.get("id")): c for c in (chunks or []) if c.get("id") is not None}

    # Relevance is summary-side precision: source token support, with a separate
    # unsupported-token penalty; this does not measure how much source was covered.
    if summary_tokens and source_tokens and (summary_language or source_language) == source_language:
        token_precision = sum(count for token, count in sum_counts.items() if token in src_set) / len(summary_tokens)
        supported_tokens = sum(count for token, count in sum_counts.items() if token in src_set)
        unsupported_ratio = 1 - token_precision
        score = 100 * token_precision
        relevance = _metric(score, f"{supported_tokens} of {len(summary_tokens)} meaningful summary tokens occur in the source ({round(score)}% lexical support). Requested purpose: {purpose or 'general summary'}; purpose intent is not semantically scored by this deterministic method.",
                            evidence=[{"method": "source-token precision", "supported_tokens": supported_tokens, "summary_tokens": len(summary_tokens), "unsupported_token_ratio": round(unsupported_ratio, 4)}])
    else:
        relevance = _metric(None, "Source and summary could not be compared reliably with the available language metadata.", status="unavailable")

    # Completeness is source-side recall over a depth-dependent set of salient
    # source sentences, weighted by distinctive terms and decision/risk/numeric cues.
    if summary_tokens and src_sents and (summary_language or source_language) == source_language:
        source_counts = Counter(source_tokens)
        max_units = {"quick": 3, "standard": 6, "deep": 12, "auto": 6, "custom": 8}.get((depth or "standard").lower(), 6)
        sentence_rows = []
        for sentence in src_sents:
            terms = _tokens(sentence)
            if not terms:
                continue
            rarity = sum(math.log1p(len(source_tokens) / source_counts[t]) for t in set(terms)) / len(set(terms))
            cue_bonus = min(2.0, len(IMPORTANT_CUES.findall(sentence)) * 0.45)
            salience = rarity + cue_bonus
            coverage = len(set(terms) & set(summary_tokens)) / max(1, len(set(terms)))
            sentence_rows.append((salience, coverage, sentence, len(terms)))
        chosen = sorted(sentence_rows, key=lambda row: row[0], reverse=True)[:min(max_units, len(sentence_rows))]
        denominator = sum(row[0] for row in chosen)
        completeness_score = 100 * sum(row[0] * row[1] for row in chosen) / denominator if denominator else None
        missing = [row[2] for row in chosen if row[1] < 0.5][:4]
        completeness = _metric(completeness_score,
            f"Coverage of {len(chosen)} salient source sentences selected for {depth or 'standard'} depth, weighted by distinctive terms and factual/risk/decision cues.",
            status="completed" if completeness_score is not None else "unavailable",
            evidence=[{"source_sentence": sentence[:360], "coverage": round(cov, 4), "salience_weight": round(weight, 4)} for weight, cov, sentence, _ in chosen[:8]])
    else:
        missing, completeness = [], _metric(None, "No comparable source sentences were available.", status="unavailable")

    # Each supplied pipeline claim remains attached to its actual source refs.
    evaluated_claims: list[dict[str, Any]] = []
    definitive = []
    for claim in claims:
        refs, evidence = _claim_refs(claim, chunks_by_id)
        status = str(claim.get("status") or "UNAVAILABLE").upper()
        record = {"claim": str(claim.get("claim") or "").strip(), "validation_result": status,
                  "source_evidence": evidence[:3], "sources": refs, "confidence": claim.get("confidence")}
        record["supported"] = True if status == "SUPPORTED" else False if status == "UNSUPPORTED" else None
        evaluated_claims.append(record)
        if status in {"SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"}:
            definitive.append((claim, status, refs, evidence))

    claim_status = {"SUPPORTED": 1.0, "PARTIALLY_SUPPORTED": 0.5, "UNSUPPORTED": 0.0}
    if definitive:
        consistency_score = 100 * sum(claim_status[status] for _, status, _, _ in definitive) / len(definitive)
        unsupported = [entry for entry in evaluated_claims if entry["validation_result"] in {"UNSUPPORTED", "PARTIALLY_SUPPORTED"}]
        factual = _metric(consistency_score, f"Pipeline claim validation classified {sum(s == 'SUPPORTED' for _, s, _, _ in definitive)} of {len(definitive)} adjudicated claims as supported; partial claims count half and unsupported claims count zero.", evidence=[{"claim": c["claim"], "validation_result": s} for c, s, _, _ in definitive[:10]])
        grounded_rows = []
        for claim, status_value, refs, evidence in definitive:
            claim_tokens = set(_tokens(str(claim.get("claim") or "")))
            cited_tokens = set(_tokens(" ".join(evidence)))
            overlap = len(claim_tokens & cited_tokens) / max(1, len(claim_tokens)) if refs and evidence else 0.0
            claim_numbers = set(re.findall(r"\d+(?:[.,]\d+)?%?", str(claim.get("claim") or "")))
            evidence_numbers = set(re.findall(r"\d+(?:[.,]\d+)?%?", " ".join(evidence)))
            if claim_numbers - evidence_numbers:
                overlap = 0.0
            grounded_rows.append((claim, status_value, refs, evidence, overlap))
        grounded_score = 100 * sum(row[4] for row in grounded_rows) / len(grounded_rows)
        grounded_claims = [item for item in evaluated_claims if item["validation_result"] == "SUPPORTED" and item["sources"] and item["source_evidence"] and next((row[4] >= 0.65 for row in grounded_rows if row[0].get("claim") == item["claim"]), False)]
        ungrounded_claims = [item for item in evaluated_claims if item not in grounded_claims]
        grounded = _metric(grounded_score, f"Mean lexical coverage of each adjudicated claim by its cited source text; invalid/missing citations and number mismatches score zero. This traceability measure is independent of the pipeline's supported/unsupported label.",
                           evidence=[{"claim": c.get("claim"), "source_coverage": round(overlap, 4), "sources": refs, "evidence": evidence[:2]} for c, _, refs, evidence, overlap in grounded_rows[:10]])
    else:
        unsupported, grounded_claims, ungrounded_claims = [], [], []
        factual = _metric(None, "No adjudicated factual claims are available to compare with the source.", status="unavailable")
        grounded = _metric(None, "No adjudicated factual claims are available for source-evidence matching.", status="unavailable")

    # Validation accuracy is an independent local audit against cited source text.
    audits = []
    for claim, label, _refs, _evidence in definitive:
        if label not in {"SUPPORTED", "UNSUPPORTED"}:
            continue
        audit_sources = _independent_source_evidence(str(claim.get("claim") or ""), chunks)
        audit_evidence = [text for _, text in audit_sources]
        audit = basic_local_claim_check(str(claim.get("claim") or ""), audit_evidence,
            claim_language=summary_language or source_language, source_language=source_language,
            cited_ids_valid=bool(audit_sources))
        if audit.get("status") in {"SUPPORTED", "UNSUPPORTED"}:
            audits.append((claim, label, audit["status"], audit.get("reason", ""),
                           [{"chunk_id": chunk.get("id"), **({"page": chunk.get("page")} if chunk.get("page") is not None else {}),
                             **({"section": chunk.get("section")} if chunk.get("section") else {})} for chunk, _ in audit_sources]))
    if not claims:
        validation_accuracy = _metric(None, "No factual claims requiring validation were detected.", status="not_applicable")
    elif audits:
        agreement = sum(label == actual for _, label, actual, _, _ in audits)
        validation_accuracy = _metric(100 * agreement / len(audits), f"Independent deterministic source-overlap audit against the highest-overlap source chunks agreed with the pipeline on {agreement} of {len(audits)} definitive claim decisions. Partial and inconclusive local checks are excluded.", evidence=[{"claim": c.get("claim"), "pipeline_result": label, "independent_local_result": actual, "basis": reason, "source_chunks": refs} for c, label, actual, reason, refs in audits[:10]])
    else:
        validation_accuracy = _metric(None, "The independent source-overlap audit could not make definitive supported/unsupported decisions for these claims.", status="unavailable")

    # Summary quality uses independent observable proxies, not source overlap:
    # sentence length, repeated content, organization, and source concision.
    if summary_tokens:
        sentence_count = max(1, len(sum_sents))
        avg_sentence = len(summary_tokens) / sentence_count
        clarity = max(0.0, 1.0 - max(0.0, avg_sentence - 24) / 36)
        unique_ratio = len(set(summary_tokens)) / len(summary_tokens)
        repetition = 1.0 - (sum(max(0, count - 2) for count in Counter(_tokens(" ".join(sum_sents))).values()) / max(1, len(summary_tokens)))
        repetition = max(0.0, min(1.0, repetition))
        organization = min(1.0, 0.55 + 0.15 * bool(re.search(r"(?m)^\s*(?:#{1,3}\s|[-*•]\s|\d+[.)]\s)", summary)) + 0.1 * (sentence_count >= 3) + 0.1 * ("\n" in summary))
        density = min(1.0, unique_ratio / 0.72)
        if source_tokens:
            ratio = len(summary_tokens) / len(source_tokens)
            concision = 1.0 if ratio <= 0.35 else max(0.0, 1.0 - (ratio - 0.35) / 0.65)
            quality_score = 100 * (0.30 * clarity + 0.20 * repetition + 0.20 * organization + 0.20 * density + 0.10 * concision)
            quality_status = "completed"
        else:
            quality_score, quality_status = None, "unavailable"
        quality = _metric(quality_score, "Observable rubric: sentence length/readability (30%), repetition (20%), organization (20%), information density (20%), and source-relative concision (10%). These are structural proxies, not a semantic style judgment.", status=quality_status,
                          evidence=[{"sentence_count": len(sum_sents), "average_meaningful_tokens_per_sentence": round(avg_sentence, 2), "unique_token_ratio": round(unique_ratio, 4), "summary_to_source_token_ratio": round(len(summary_tokens) / max(1, len(source_tokens)), 4)}])
    else:
        quality = _metric(None, "The summary contains no measurable content.", status="unavailable")

    metrics = {"relevance": relevance, "completeness": completeness,
               "factual_consistency": factual, "groundedness": grounded,
               "summary_quality": quality, "validation_accuracy": validation_accuracy}
    applicable = [(metrics[name]["score"], weight) for name, weight in WEIGHTS.items() if metrics[name]["score"] is not None]
    overall = sum(score * weight for score, weight in applicable) / sum(weight for _, weight in applicable) if applicable else None
    if not applicable:
        status = "failed"
    elif validation_accuracy["status"] == "not_applicable":
        status = "not_applicable"
    elif any(metrics[name]["score"] is None and metrics[name]["status"] != "not_applicable" for name in METRIC_NAMES):
        status = "partial"
    else:
        status = "completed"
    warnings = []
    if (summary_language or source_language) != source_language:
        warnings.append("Summary language differs from the source; lexical relevance and completeness were not scored.")
    if provider:
        provider_used = provider.get("name") or "unknown"
    else:
        provider_used = "programmatic"
    return {
        "evaluation_status": status, "reason": None if status != "failed" else "No metric could be calculated from the available source and summary.",
        "metrics": metrics, "overall_score": round(overall, 2) if overall is not None else None,
        "overall_score_rounded": round(overall) if overall is not None else None,
        "applicable_metric_count": len(applicable), "total_metric_count": 6,
        "overall_formula": {"weights": WEIGHTS, "rule": "weighted mean of available metric scores; N/A and unavailable metrics are excluded and remaining weights normalized"},
        "claims": evaluated_claims, "grounded_claims": grounded_claims,
        "ungrounded_claims": ungrounded_claims, "unsupported_claims": unsupported,
        "important_source_coverage": {"selected_units": len(chosen) if 'chosen' in locals() else 0, "missing_units": missing},
        "warnings": warnings, "evaluator": {"provider": "programmatic", "model": EVALUATION_VERSION, "summary_provider": provider_used},
        "evaluation_version": EVALUATION_VERSION, "summary_fingerprint": fingerprint,
    }
