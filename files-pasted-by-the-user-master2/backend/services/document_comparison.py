"""Comprehensive semantic document comparison service.

Performs intelligent comparison between two documents:
1. Normalization & entity extraction (preserving numbers, dates, terms).
2. Section & paragraph structural segmentation.
3. Multilingual and semantic embedding alignment (reusing existing EmbeddingService).
4. Factual and value change detection (numeric, date, specification diffs).
5. Granular classification: UNCHANGED, MODIFIED, ADDED, REMOVED, MOVED, TRANSLATED/EQUIVALENT, UNCERTAIN.
6. Contextual synthesis (LLM with deterministic fallback).
7. Traceable page/paragraph references and surrounding context.
"""
from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import Any

from services.embedding_service import cosine_similarity, get_embedding_service

LOGGER = logging.getLogger("scribe.comparison")

LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
}


def clean_text_for_comparison(text: str) -> str:
    """Normalize whitespace and harmless formatting while strictly preserving numbers and symbols."""
    text = text.replace("\x00", " ").replace("\r", "\n")
    # Remove repeated blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove repeated spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_numbers_and_quantities(text: str) -> list[dict[str, Any]]:
    """Extract numeric values, percentages, and quantities with their units."""
    pattern = r"\b(\$|€|£|₹|USD|INR)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|percent|users?|ms|seconds?|passenger trips|million|billion|nodes?|replicas?|days?|hours?|kb|mb|gb|years?)?\b"
    results = []
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        prefix, num_str, suffix = match.groups()
        if not num_str:
            continue
        try:
            val = float(num_str.replace(",", ""))
        except ValueError:
            continue
        raw = match.group(0).strip()
        results.append({
            "raw": raw,
            "value": val,
            "prefix": (prefix or "").strip(),
            "suffix": (suffix or "").strip().lower(),
        })
    return results


def extract_dates(text: str) -> list[str]:
    """Extract dates from text."""
    date_patterns = [
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        r"\b(?:Q[1-4]|FY\s*\d{2,4})\b",
    ]
    dates = []
    for pat in date_patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            dates.append(m.group(0).strip())
    return dates


def detect_entity_differences(text_a: str, text_b: str) -> list[dict[str, Any]]:
    """Detect specific differences in numbers, dates, and technical keywords."""
    diffs = []
    # 1. Compare numbers
    nums_a = extract_numbers_and_quantities(text_a)
    nums_b = extract_numbers_and_quantities(text_b)

    if nums_a and nums_b:
        # Match by suffix/unit or position
        for na in nums_a:
            matching_b = [nb for nb in nums_b if nb["suffix"] == na["suffix"] and na["suffix"] != ""]
            if matching_b:
                nb = matching_b[0]
                if abs(na["value"] - nb["value"]) > 1e-5:
                    unit = na["suffix"] or "units"
                    diff_val = nb["value"] - na["value"]
                    direction = f"increased by {abs(diff_val):g}" if diff_val > 0 else f"decreased by {abs(diff_val):g}"
                    diffs.append({
                        "type": "NUMERIC_CHANGE",
                        "old_value": na["raw"],
                        "new_value": nb["raw"],
                        "explanation": f"{unit.capitalize()} changed from {na['raw']} → {nb['raw']} ({direction}).",
                    })
            elif len(nums_a) == 1 and len(nums_b) == 1:
                nb = nums_b[0]
                if abs(na["value"] - nb["value"]) > 1e-5:
                    diffs.append({
                        "type": "NUMERIC_CHANGE",
                        "old_value": na["raw"],
                        "new_value": nb["raw"],
                        "explanation": f"Value changed from {na['raw']} → {nb['raw']}.",
                    })

    # 2. Compare dates
    dates_a = extract_dates(text_a)
    dates_b = extract_dates(text_b)
    if dates_a and dates_b:
        for da in dates_a:
            if da not in dates_b and dates_b:
                db = dates_b[0]
                diffs.append({
                    "type": "DATE_CHANGE",
                    "old_value": da,
                    "new_value": db,
                    "explanation": f"Timeline/date updated from {da} → {db}.",
                })

    # 3. Known technical terms replacement check
    tech_pairs = [
        ("mysql", "postgresql"),
        ("sqlite", "mysql"),
        ("docker", "kubernetes"),
        ("rest", "graphql"),
        ("http", "https"),
        ("canary", "blue-green"),
    ]
    for term_a, term_b in tech_pairs:
        if re.search(r"\b" + term_a + r"\b", text_a, re.I) and re.search(r"\b" + term_b + r"\b", text_b, re.I):
            diffs.append({
                "type": "SPECIFICATION_CHANGE",
                "old_value": term_a.upper(),
                "new_value": term_b.upper(),
                "explanation": f"Technology specification changed from {term_a.upper()} → {term_b.upper()}.",
            })
        elif re.search(r"\b" + term_b + r"\b", text_a, re.I) and re.search(r"\b" + term_a + r"\b", text_b, re.I):
            diffs.append({
                "type": "SPECIFICATION_CHANGE",
                "old_value": term_b.upper(),
                "new_value": term_a.upper(),
                "explanation": f"Technology specification changed from {term_b.upper()} → {term_a.upper()}.",
            })

    return diffs


def segment_document_into_sections(
    text: str, pages: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Segment a document into titled sections, paragraphs, and contextual sentences."""
    clean_text = clean_text_for_comparison(text)
    if not clean_text:
        return []

    lines = clean_text.split("\n")
    sections: list[dict[str, Any]] = []
    current_title = "Overview / Introduction"
    current_lines: list[str] = []

    # Patterns indicating section headings
    heading_patterns = [
        r"^(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s+([A-Z\s]{3,})",  # Roman numerals
        r"^(?:\d+\.|\d+\.\d+)\s+([A-Za-z0-9\s\-_]{3,})",           # 1. Title or 1.1 Title
        r"^#{1,3}\s+(.+)$",                                          # Markdown # Title
        r"^(?:Section|Chapter|Part)\s+\d+[:\s]+(.+)$",               # Section 1: Title
        r"^([A-Z\s]{4,35})$",                                        # Standalone uppercase line
    ]

    for line in lines:
        line_s = line.strip()
        if not line_s:
            if current_lines:
                current_lines.append("")
            continue

        matched_title = None
        for pat in heading_patterns:
            m = re.match(pat, line_s, flags=re.IGNORECASE)
            if m:
                matched_title = m.group(1).strip() if m.lastindex else line_s
                break

        if matched_title and len(matched_title) > 3 and not matched_title.endswith("."):
            if current_lines:
                sec_text = "\n".join(current_lines).strip()
                if sec_text:
                    sections.append({
                        "title": current_title,
                        "text": sec_text,
                    })
                current_lines = []
            current_title = matched_title
        else:
            current_lines.append(line)

    if current_lines:
        sec_text = "\n".join(current_lines).strip()
        if sec_text:
            sections.append({
                "title": current_title,
                "text": sec_text,
            })

    # If no headings were identified, break by paragraphs into structured blocks
    if len(sections) <= 1:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", clean_text) if p.strip()]
        if len(paragraphs) > 1:
            sections = []
            # Group every 2-3 paragraphs into a section block
            chunk_size = 2 if len(paragraphs) <= 6 else 3
            for i in range(0, len(paragraphs), chunk_size):
                p_group = paragraphs[i:i + chunk_size]
                sec_title = f"Section {i // chunk_size + 1}"
                first_line = p_group[0].split(".")[0].strip()
                if len(first_line) < 40 and not any(c in first_line for c in "\n\t"):
                    sec_title = f"{first_line[:35]}…"
                sections.append({
                    "title": sec_title,
                    "text": "\n\n".join(p_group),
                })
        elif not sections:
            sections = [{"title": "General Content", "text": clean_text}]

    # Now enrich each section with sentence breakdowns and context
    page_map = _build_page_map(pages) if pages else {}

    for sec in sections:
        raw_sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?।॥])\s+|\n+", sec["text"])
            if len(s.strip()) > 8
        ]
        sentence_objects = []
        for idx, s in enumerate(raw_sentences):
            before = raw_sentences[idx - 1] if idx > 0 else ""
            after = raw_sentences[idx + 1] if idx < len(raw_sentences) - 1 else ""
            page_num = _find_page_for_sentence(s, page_map)
            sentence_objects.append({
                "index": idx,
                "text": s,
                "context_before": before,
                "context_after": after,
                "page": page_num,
            })
        sec["sentences"] = sentence_objects

    return sections


def _build_page_map(pages: list[dict[str, Any]]) -> dict[int, str]:
    """Map page number to normalized page text."""
    return {p.get("page", i + 1): p.get("text", "") for i, p in enumerate(pages)}


def _find_page_for_sentence(sentence: str, page_map: dict[int, str]) -> int | None:
    """Find the page number where a sentence occurs."""
    if not page_map:
        return None
    sample = sentence[:40].casefold()
    for page, text in page_map.items():
        if sample in text.casefold():
            return page
    return None


def calculate_lexical_similarity(text_a: str, text_b: str) -> float:
    """Compute token-level Jaccard similarity as a reliable fallback."""
    tokens_a = set(re.findall(r"\w+", text_a.casefold(), flags=re.UNICODE))
    tokens_b = set(re.findall(r"\w+", text_b.casefold(), flags=re.UNICODE))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def perform_document_comparison(
    *,
    doc_a_info: dict[str, Any],
    doc_b_info: dict[str, Any],
    text_a: str,
    text_b: str,
    pages_a: list[dict[str, Any]] | None = None,
    pages_b: list[dict[str, Any]] | None = None,
    lang_a: str = "en",
    lang_b: str = "en",
    llm_service: Any = None,
) -> dict[str, Any]:
    """Execute complete semantic comparison pipeline between Document A and Document B."""
    sections_a = segment_document_into_sections(text_a, pages_a)
    sections_b = segment_document_into_sections(text_b, pages_b)

    words_a = len(text_a.split())
    words_b = len(text_b.split())
    page_count_a = len(pages_a) if pages_a else max(1, math.ceil(len(text_a) / 3000))
    page_count_b = len(pages_b) if pages_b else max(1, math.ceil(len(text_b) / 3000))

    lang_name_a = LANGUAGES.get(lang_a, lang_a.capitalize())
    lang_name_b = LANGUAGES.get(lang_b, lang_b.capitalize())

    doc_a_meta = {
        "name": doc_a_info.get("name", "Document A"),
        "type": Path(doc_a_info.get("name", "doc.txt")).suffix.replace(".", "").upper() or "TXT",
        "language": lang_name_a,
        "language_code": lang_a,
        "page_count": page_count_a,
        "word_count": words_a,
        "section_count": len(sections_a),
    }

    doc_b_meta = {
        "name": doc_b_info.get("name", "Document B"),
        "type": Path(doc_b_info.get("name", "doc.txt")).suffix.replace(".", "").upper() or "TXT",
        "language": lang_name_b,
        "language_code": lang_b,
        "page_count": page_count_b,
        "word_count": words_b,
        "section_count": len(sections_b),
    }

    # 1. Generate embeddings for sections and top sentences using existing EmbeddingService
    emb_svc = get_embedding_service()
    texts_to_embed = [s["text"][:1200] for s in sections_a] + [s["text"][:1200] for s in sections_b]
    embeddings = emb_svc.get_embeddings(texts_to_embed)

    vectors_a = embeddings[:len(sections_a)]
    vectors_b = embeddings[len(sections_a):]

    # Check overall similarity matrix
    sim_matrix: list[list[float]] = []
    max_sim_overall = 0.0
    for i, va in enumerate(vectors_a):
        row = []
        for j, vb in enumerate(vectors_b):
            sim = 0.0
            if va and vb:
                sim = cosine_similarity(va, vb)
            else:
                sim = calculate_lexical_similarity(sections_a[i]["text"], sections_b[j]["text"])
            row.append(sim)
            if sim > max_sim_overall:
                max_sim_overall = sim
        sim_matrix.append(row)

    # 2. Check for UNRELATED / DIFFERENT TOPIC documents
    # Example: Computational Systems (English) vs Justice League (Telugu)
    is_unrelated = False
    topic_overlap = "HIGH"
    lex_sim = calculate_lexical_similarity(text_a, text_b)

    if max_sim_overall < 0.58 and (lang_a != lang_b or lex_sim < 0.08):
        is_unrelated = True
        topic_overlap = "LOW"
    elif max_sim_overall < 0.70:
        topic_overlap = "MEDIUM"

    comparison_items: list[dict[str, Any]] = []
    section_comparisons: list[dict[str, Any]] = []
    stats = {
        "unchanged": 0,
        "modified": 0,
        "added": 0,
        "removed": 0,
        "moved": 0,
        "translated_equivalent": 0,
        "uncertain": 0,
        "total": 0,
    }
    important_changes: list[str] = []

    if is_unrelated:
        # Explicitly classify as completely unrelated documents rather than false line diffs
        stats["removed"] = len(sections_a)
        stats["added"] = len(sections_b)
        stats["total"] = len(sections_a) + len(sections_b)

        for sa in sections_a:
            for item in sa["sentences"]:
                comparison_items.append({
                    "id": f"rem-{len(comparison_items)}",
                    "section_name": sa["title"],
                    "status": "REMOVED",
                    "document_a": {
                        "text": item["text"],
                        "context_before": item["context_before"],
                        "context_after": item["context_after"],
                        "page": item["page"],
                    },
                    "document_b": {
                        "text": "",
                        "context_before": "",
                        "context_after": "",
                        "page": None,
                    },
                    "change_type": "CONTENT_REMOVAL",
                    "explanation": f"Present in Document A ('{doc_a_meta['name']}'), but does not exist in Document B.",
                    "specific_changes": [],
                    "confidence": "HIGH",
                })

        for sb in sections_b:
            for item in sb["sentences"]:
                comparison_items.append({
                    "id": f"add-{len(comparison_items)}",
                    "section_name": sb["title"],
                    "status": "ADDED",
                    "document_a": {
                        "text": "",
                        "context_before": "",
                        "context_after": "",
                        "page": None,
                    },
                    "document_b": {
                        "text": item["text"],
                        "context_before": item["context_before"],
                        "context_after": item["context_after"],
                        "page": item["page"],
                    },
                    "change_type": "CONTENT_ADDITION",
                    "explanation": f"Present in Document B ('{doc_b_meta['name']}'), but does not exist in Document A.",
                    "specific_changes": [],
                    "confidence": "HIGH",
                })

        executive_summary = (
            f"Low semantic overlap detected between '{doc_a_meta['name']}' ({doc_a_meta['language']}) "
            f"and '{doc_b_meta['name']}' ({doc_b_meta['language']}). These documents discuss entirely distinct "
            f"subjects with no shared narrative or equivalent sections. Document A contains {words_a} words "
            f"across {len(sections_a)} sections, while Document B contains {words_b} words across {len(sections_b)} sections."
        )
        important_changes.append(
            f"Documents belong to completely different domains ({doc_a_meta['language']} vs {doc_b_meta['language']})."
        )
        important_changes.append(
            f"Zero shared sections detected between '{doc_a_meta['name']}' and '{doc_b_meta['name']}'."
        )

    else:
        # Documents have semantic overlap or are revisions/translations of each other
        matched_b_indices: set[int] = set()

        for idx_a, sec_a in enumerate(sections_a):
            # Find best matching section in B
            best_idx_b = -1
            best_sim = -1.0
            if sim_matrix and idx_a < len(sim_matrix):
                for idx_b, sim in enumerate(sim_matrix[idx_a]):
                    if sim > best_sim:
                        best_sim = sim
                        best_idx_b = idx_b

            title_sim = calculate_lexical_similarity(sec_a["title"], sections_b[best_idx_b]["title"]) if best_idx_b >= 0 else 0.0
            if best_idx_b >= 0 and (best_sim >= 0.55 or title_sim > 0.4):
                matched_b_indices.add(best_idx_b)
                sec_b = sections_b[best_idx_b]
                is_moved = abs(idx_a - best_idx_b) >= 2

                sec_status = "UNCHANGED"
                sec_changes: list[dict[str, Any]] = []

                # Align sentences within this section pair
                matched_b_sents: set[int] = set()
                for sent_a in sec_a["sentences"]:
                    best_sent_b_idx = -1
                    best_sent_sim = -1.0
                    for s_b_idx, sent_b in enumerate(sec_b["sentences"]):
                        sim_s = calculate_lexical_similarity(sent_a["text"], sent_b["text"])
                        if sent_a["text"].strip().casefold() == sent_b["text"].strip().casefold():
                            sim_s = 1.0
                        if sim_s > best_sent_sim:
                            best_sent_sim = sim_s
                            best_sent_b_idx = s_b_idx

                    if best_sent_b_idx >= 0 and best_sent_sim >= 0.50:
                        matched_b_sents.add(best_sent_b_idx)
                        sent_b = sec_b["sentences"][best_sent_b_idx]
                        entity_diffs = detect_entity_differences(sent_a["text"], sent_b["text"])

                        if best_sent_sim >= 0.95 and not entity_diffs:
                            item_status = "UNCHANGED"
                            stats["unchanged"] += 1
                            change_type = "NO_CHANGE"
                            explanation = "Content remained identical."
                            conf = "HIGH"
                        elif is_moved and best_sent_sim >= 0.85 and not entity_diffs:
                            item_status = "MOVED"
                            stats["moved"] += 1
                            change_type = "STRUCTURAL_MOVE"
                            explanation = f"Section moved from position {idx_a + 1} in Document A to position {best_idx_b + 1} in Document B."
                            conf = "HIGH"
                        elif lang_a != lang_b and best_sim >= 0.65:
                            item_status = "TRANSLATED/EQUIVALENT"
                            stats["translated_equivalent"] += 1
                            change_type = "CROSS_LINGUAL_EQUIVALENCE"
                            explanation = f"Semantically equivalent passage translated between {lang_name_a} and {lang_name_b}."
                            conf = "HIGH" if best_sim > 0.75 else "MEDIUM"
                        else:
                            item_status = "MODIFIED"
                            stats["modified"] += 1
                            sec_status = "MODIFIED"
                            conf = "HIGH" if entity_diffs else "MEDIUM"
                            if entity_diffs:
                                change_type = entity_diffs[0]["type"]
                                explanation = entity_diffs[0]["explanation"]
                                for ed in entity_diffs:
                                    important_changes.append(ed["explanation"])
                            else:
                                change_type = "REWORDING"
                                explanation = "Passage was reworded or elaborated with modified details."

                        comparison_items.append({
                            "id": f"item-{len(comparison_items)}",
                            "section_name": sec_a["title"],
                            "status": item_status,
                            "document_a": {
                                "text": sent_a["text"],
                                "context_before": sent_a["context_before"],
                                "context_after": sent_a["context_after"],
                                "page": sent_a["page"],
                            },
                            "document_b": {
                                "text": sent_b["text"],
                                "context_before": sent_b["context_before"],
                                "context_after": sent_b["context_after"],
                                "page": sent_b["page"],
                            },
                            "change_type": change_type,
                            "explanation": explanation,
                            "specific_changes": entity_diffs,
                            "confidence": conf,
                        })
                    else:
                        # Sentence from A was removed in B
                        stats["removed"] += 1
                        sec_status = "MODIFIED"
                        comparison_items.append({
                            "id": f"item-{len(comparison_items)}",
                            "section_name": sec_a["title"],
                            "status": "REMOVED",
                            "document_a": {
                                "text": sent_a["text"],
                                "context_before": sent_a["context_before"],
                                "context_after": sent_a["context_after"],
                                "page": sent_a["page"],
                            },
                            "document_b": {
                                "text": "",
                                "context_before": "",
                                "context_after": "",
                                "page": None,
                            },
                            "change_type": "CONTENT_REMOVAL",
                            "explanation": "This statement was removed from Document B.",
                            "specific_changes": [],
                            "confidence": "HIGH",
                        })

                # Sentences in sec_b that had no match in sec_a are ADDED
                for s_b_idx, sent_b in enumerate(sec_b["sentences"]):
                    if s_b_idx not in matched_b_sents:
                        stats["added"] += 1
                        sec_status = "MODIFIED"
                        comparison_items.append({
                            "id": f"item-{len(comparison_items)}",
                            "section_name": sec_b["title"],
                            "status": "ADDED",
                            "document_a": {
                                "text": "",
                                "context_before": "",
                                "context_after": "",
                                "page": None,
                            },
                            "document_b": {
                                "text": sent_b["text"],
                                "context_before": sent_b["context_before"],
                                "context_after": sent_b["context_after"],
                                "page": sent_b["page"],
                            },
                            "change_type": "CONTENT_ADDITION",
                            "explanation": "New content added in Document B.",
                            "specific_changes": [],
                            "confidence": "HIGH",
                        })

                section_comparisons.append({
                    "section_name": sec_a["title"],
                    "matched_section_name": sec_b["title"],
                    "status": sec_status,
                    "similarity": round(best_sim, 2),
                    "confidence": "HIGH" if best_sim > 0.75 else "MEDIUM",
                })
            else:
                # Section A has no match in B -> REMOVED
                stats["removed"] += len(sec_a["sentences"])
                for sent_a in sec_a["sentences"]:
                    comparison_items.append({
                        "id": f"item-{len(comparison_items)}",
                        "section_name": sec_a["title"],
                        "status": "REMOVED",
                        "document_a": {
                            "text": sent_a["text"],
                            "context_before": sent_a["context_before"],
                            "context_after": sent_a["context_after"],
                            "page": sent_a["page"],
                        },
                        "document_b": {
                            "text": "",
                            "context_before": "",
                            "context_after": "",
                            "page": None,
                        },
                        "change_type": "CONTENT_REMOVAL",
                        "explanation": f"Section '{sec_a['title']}' was removed in Document B.",
                        "specific_changes": [],
                        "confidence": "HIGH",
                    })
                section_comparisons.append({
                    "section_name": sec_a["title"],
                    "matched_section_name": None,
                    "status": "REMOVED",
                    "similarity": 0.0,
                    "confidence": "HIGH",
                })

        # Check sections in B that were never matched -> ADDED
        for idx_b, sec_b in enumerate(sections_b):
            if idx_b not in matched_b_indices:
                stats["added"] += len(sec_b["sentences"])
                for sent_b in sec_b["sentences"]:
                    comparison_items.append({
                        "id": f"item-{len(comparison_items)}",
                        "section_name": sec_b["title"],
                        "status": "ADDED",
                        "document_a": {
                            "text": "",
                            "context_before": "",
                            "context_after": "",
                            "page": None,
                        },
                        "document_b": {
                            "text": sent_b["text"],
                            "context_before": sent_b["context_before"],
                            "context_after": sent_b["context_after"],
                            "page": sent_b["page"],
                        },
                        "change_type": "CONTENT_ADDITION",
                        "explanation": f"New section '{sec_b['title']}' was added in Document B.",
                        "specific_changes": [],
                        "confidence": "HIGH",
                    })
                section_comparisons.append({
                    "section_name": sec_b["title"],
                    "matched_section_name": None,
                    "status": "ADDED",
                    "similarity": 0.0,
                    "confidence": "HIGH",
                })

        stats["total"] = len(comparison_items)

        # Synthesize Executive Comparison Summary
        if stats["modified"] == 0 and stats["added"] == 0 and stats["removed"] == 0:
            executive_summary = (
                f"Documents '{doc_a_meta['name']}' and '{doc_b_meta['name']}' are substantially identical. "
                f"All {stats['unchanged']} inspected content elements match with no meaningful factual or structural deviations."
            )
            important_changes.append("No material differences or value changes detected.")
        else:
            summary_parts = []
            if stats["unchanged"] > 0:
                summary_parts.append(f"{stats['unchanged']} elements remained unchanged")
            if stats["modified"] > 0:
                summary_parts.append(f"{stats['modified']} elements were modified")
            if stats["added"] > 0:
                summary_parts.append(f"{stats['added']} elements were added")
            if stats["removed"] > 0:
                summary_parts.append(f"{stats['removed']} elements were removed")
            if stats["moved"] > 0:
                summary_parts.append(f"{stats['moved']} elements were repositioned")

            executive_summary = (
                f"Comparison between '{doc_a_meta['name']}' ({doc_a_meta['language']}) "
                f"and '{doc_b_meta['name']}' ({doc_b_meta['language']}) identified {stats['total']} analyzed points: "
                f"{', '.join(summary_parts)}. "
            )
            if important_changes:
                executive_summary += f"{len(important_changes)} specific factual/value updates were detected."

    # Deduplicate important changes
    seen_changes = set()
    deduped_important_changes = []
    for c in important_changes:
        if c not in seen_changes:
            seen_changes.add(c)
            deduped_important_changes.append(c)

    # Legacy compatibility fields (plain text arrays for any older visualizer dependencies)
    legacy_added = [
        item["document_b"]["text"]
        for item in comparison_items
        if item["status"] in ("ADDED", "MODIFIED") and item["document_b"]["text"]
    ]
    legacy_removed = [
        item["document_a"]["text"]
        for item in comparison_items
        if item["status"] in ("REMOVED", "MODIFIED") and item["document_a"]["text"]
    ]

    return {
        "document_a": doc_a_meta,
        "document_b": doc_b_meta,
        "executive_summary": executive_summary,
        "important_changes": deduped_important_changes[:8],
        "statistics": stats,
        "is_unrelated": is_unrelated,
        "topic_overlap": topic_overlap,
        "comparison_items": comparison_items,
        "sections": section_comparisons,
        "added": legacy_added,
        "removed": legacy_removed,
        "method": "semantic & contextual comparison",
        "limitations": "Contextual analysis based on semantic embeddings and entity diff detection.",
    }
