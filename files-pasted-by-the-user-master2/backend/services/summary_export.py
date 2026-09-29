"""Render persisted Scribe summaries as downloadable documents."""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import io
import json
from typing import Any


SUPPORTED_EXPORT_FORMATS = {"pdf", "docx", "txt", "md", "markdown", "json"}
MAX_EXPORT_CHARACTERS = 2_000_000

_SECTION_FIELDS = (
    ("findings", "key_findings"),
    ("facts", "important_facts"),
    ("main_points", "main_points"),
    ("risks", "risks"),
    ("decisions", "decisions"),
    ("recommendations", "recommendations"),
    ("action_items", "action_items"),
    ("conclusion", "conclusion"),
)


class SummaryNotAvailable(ValueError):
    pass


class ExportTooLarge(ValueError):
    pass


def make_export_payload(document: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    summary = str(result.get("executive_summary") or "").strip()
    if not summary:
        raise SummaryNotAvailable("A summary is not available for this document yet.")
    name = str(document.get("name") or "Document")
    payload: dict[str, Any] = {
        "document": name,
        "original_file_name": name,
        "language": result.get("language") or document.get("language") or "und",
        "summary": summary,
        "summary_depth": result.get("summary_depth_resolved") or result.get("depth"),
        "generated_at": result.get("generated_at") or datetime.now(timezone.utc).isoformat(),
    }
    validation_state = result.get("validation_state")
    if validation_state:
        payload["validation_state"] = validation_state
    if result.get("validation_warning") and result.get("warning_message"):
        payload["validation_warning"] = str(result["warning_message"])
    if isinstance(result.get("quality"), dict):
        payload["supported_claims"] = int(result.get("supported_claims_count", 0) or 0)
        payload["unverified_claims"] = int(result.get("unsupported_claims_count", 0) or 0)
    for source_key, export_key in _SECTION_FIELDS:
        value = result.get(source_key)
        if isinstance(value, str) and value.strip():
            payload[export_key] = value.strip()
        elif isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
            if items:
                payload[export_key] = items
    _check_size(payload)
    return payload


def make_combined_export_payload(combined: dict[str, Any]) -> dict[str, Any]:
    summary = str(combined.get("summary") or "").strip()
    if not summary:
        raise SummaryNotAvailable("A combined summary is not available.")
    payload: dict[str, Any] = {
        "document": "Combined summary",
        "original_file_name": "Combined summary",
        "language": combined.get("language") or "und",
        "summary": summary,
        "generated_at": combined.get("generated_at") or datetime.now(timezone.utc).isoformat(),
    }
    sections = combined.get("sections")
    if isinstance(sections, list):
        items = [section for section in sections if isinstance(section, dict) and str(section.get("summary") or "").strip()]
        if items:
            payload["documents_included"] = items
    claims = combined.get("claims")
    if isinstance(claims, list):
        findings = [str(claim.get("claim") or "").strip() for claim in claims
                    if isinstance(claim, dict) and str(claim.get("claim") or "").strip()]
        if findings:
            payload["key_findings"] = findings
    _check_size(payload)
    return payload


def _check_size(payload: dict[str, Any]) -> None:
    if len(json.dumps(payload, ensure_ascii=False)) > MAX_EXPORT_CHARACTERS:
        raise ExportTooLarge("This summary is too large to export. Reduce its size and try again.")


def render_summary(payload: dict[str, Any], export_format: str) -> tuple[bytes, str, str]:
    normalized = "md" if export_format == "markdown" else export_format
    if normalized not in {"pdf", "docx", "txt", "md", "json"}:
        raise ValueError("Unsupported export format. Choose PDF, DOCX, TXT, Markdown, or JSON.")
    if normalized == "json":
        return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), "application/json; charset=utf-8", "json"
    if normalized == "md":
        return _render_markdown(payload).encode("utf-8"), "text/markdown; charset=utf-8", "md"
    if normalized == "txt":
        return _render_text(payload).encode("utf-8"), "text/plain; charset=utf-8", "txt"
    if normalized == "docx":
        return _render_docx(payload), "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"
    return _render_pdf(payload), "application/pdf", "pdf"


def _display_label(key: str) -> str:
    return key.replace("_", " ").strip().title()


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [f"# {payload['document']}", "", f"**Original file:** {payload['original_file_name']}"]
    if payload.get("language"):
        lines.append(f"**Language:** {payload['language']}")
    if payload.get("summary_depth"):
        lines.append(f"**Summary depth:** {payload['summary_depth']}")
    if payload.get("validation_state"):
        lines.append(f"**Validation:** {payload['validation_state']}")
    if payload.get("validation_warning"):
        lines.extend(["", f"> **Validation warning:** {payload['validation_warning']}"])
    if "supported_claims" in payload:
        lines.append(f"**Claims verified:** {payload['supported_claims']} · **Unverified:** {payload.get('unverified_claims', 0)}")
    lines.extend([f"**Generated:** {payload['generated_at']}", "", "## Summary", "", payload["summary"]])
    for key, value in payload.items():
        if key in {"document", "original_file_name", "language", "summary_depth", "generated_at", "summary", "validation_state", "validation_warning", "supported_claims", "unverified_claims"}:
            continue
        label = _display_label(key)
        if key == "documents_included" and isinstance(value, list):
            lines.extend(["", f"## {label}"])
            for section in value:
                lines.extend(["", f"### {section.get('document_name') or section.get('name') or 'Document'}", "", str(section.get("summary") or "")])
                for claim in section.get("findings", []) or []:
                    lines.append(f"- {claim}")
        elif isinstance(value, list) and value:
            lines.extend(["", f"## {label}", "", *[f"- {item}" for item in value]])
        elif isinstance(value, str) and value.strip():
            lines.extend(["", f"## {label}", "", value.strip()])
    return "\n".join(lines).rstrip() + "\n"


def _render_text(payload: dict[str, Any]) -> str:
    lines = [payload["document"], "=" * min(len(payload["document"]), 72),
             f"Original file: {payload['original_file_name']}"]
    if payload.get("language"):
        lines.append(f"Language: {payload['language']}")
    if payload.get("summary_depth"):
        lines.append(f"Summary depth: {payload['summary_depth']}")
    if payload.get("validation_state"):
        lines.append(f"Validation: {payload['validation_state']}")
    if payload.get("validation_warning"):
        lines.extend(["", f"VALIDATION WARNING: {payload['validation_warning']}"])
    if "supported_claims" in payload:
        lines.append(f"Claims verified: {payload['supported_claims']} | Unverified: {payload.get('unverified_claims', 0)}")
    lines.extend([f"Generated: {payload['generated_at']}", "", "Summary", "-------", payload["summary"]])
    for key, value in payload.items():
        if key in {"document", "original_file_name", "language", "summary_depth", "generated_at", "summary", "validation_state", "validation_warning", "supported_claims", "unverified_claims"}:
            continue
        lines.extend(["", _display_label(key), "-" * min(len(_display_label(key)), 72)])
        if key == "documents_included" and isinstance(value, list):
            for section in value:
                lines.extend(["", str(section.get("document_name") or section.get("name") or "Document"),
                              str(section.get("summary") or "")])
                lines.extend([f"• {item}" for item in section.get("findings", []) or []])
        elif isinstance(value, list):
            lines.extend(f"• {item}" for item in value)
        elif isinstance(value, str) and value.strip():
            lines.append(value.strip())
    return "\n".join(lines).rstrip() + "\n"


def _render_docx(payload: dict[str, Any]) -> bytes:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor

    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.7)
    normal = document.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10)
    normal.font.color.rgb = RGBColor(38, 52, 45)
    brand = document.add_paragraph()
    brand.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    brand_run = brand.add_run("SCRIBE  |  DOCUMENT INTELLIGENCE")
    brand_run.bold = True
    brand_run.font.size = Pt(9)
    brand_run.font.color.rgb = RGBColor(35, 126, 79)
    document.add_heading(payload["document"], level=0)
    metadata = [f"Original file: {payload['original_file_name']}"]
    if payload.get("language"):
        metadata.append(f"Language: {payload['language']}")
    if payload.get("summary_depth"):
        metadata.append(f"Summary depth: {payload['summary_depth']}")
    if payload.get("validation_state"):
        metadata.append(f"Validation: {payload['validation_state']}")
    if payload.get("validation_warning"):
        metadata.append(f"VALIDATION WARNING: {payload['validation_warning']}")
    if "supported_claims" in payload:
        metadata.append(f"Claims verified: {payload['supported_claims']} | Unverified: {payload.get('unverified_claims', 0)}")
    metadata.append(f"Generated: {payload['generated_at']}")
    metadata_paragraph = document.add_paragraph("\n".join(metadata))
    metadata_paragraph.style = document.styles["Subtitle"]
    document.add_heading("Summary", level=1)
    document.add_paragraph(payload["summary"])
    for key, value in payload.items():
        if key in {"document", "original_file_name", "language", "summary_depth", "generated_at", "summary", "validation_state", "validation_warning", "supported_claims", "unverified_claims"}:
            continue
        document.add_heading(_display_label(key), level=1)
        if key == "documents_included" and isinstance(value, list):
            for item in value:
                document.add_heading(str(item.get("document_name") or item.get("name") or "Document"), level=2)
                if str(item.get("summary") or "").strip():
                    document.add_paragraph(str(item["summary"]))
                for finding in item.get("findings", []) or []:
                    document.add_paragraph(str(finding), style="List Bullet")
        elif isinstance(value, list):
            for item in value:
                document.add_paragraph(str(item), style="List Bullet")
        elif isinstance(value, str) and value.strip():
            document.add_paragraph(value.strip())
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()


def _render_pdf(payload: dict[str, Any]) -> bytes:
    import fitz

    pieces = [f"<p class='brand'>SCRIBE &nbsp;|&nbsp; DOCUMENT INTELLIGENCE</p>",
              f"<h1>{escape(payload['document'])}</h1>",
              f"<p class='meta'><b>Original file:</b> {escape(payload['original_file_name'])}</p>"]
    if payload.get("language"):
        pieces.append(f"<p class='meta'><b>Language:</b> {escape(str(payload['language']))}</p>")
    if payload.get("summary_depth"):
        pieces.append(f"<p class='meta'><b>Summary depth:</b> {escape(str(payload['summary_depth']))}</p>")
    if payload.get("validation_state"):
        pieces.append(f"<p class='meta'><b>Validation:</b> {escape(str(payload['validation_state']))}</p>")
    if payload.get("validation_warning"):
        pieces.append(f"<p class='warning'><b>VALIDATION WARNING:</b> {escape(str(payload['validation_warning']))}</p>")
    if "supported_claims" in payload:
        pieces.append(f"<p class='meta'><b>Claims verified:</b> {int(payload['supported_claims'])} | <b>Unverified:</b> {int(payload.get('unverified_claims', 0))}</p>")
    pieces.extend([f"<p class='meta'><b>Generated:</b> {escape(str(payload['generated_at']))}</p>",
                   "<h2>Summary</h2>", f"<p>{escape(payload['summary']).replace(chr(10), '<br>')}</p>"])
    for key, value in payload.items():
        if key in {"document", "original_file_name", "language", "summary_depth", "generated_at", "summary", "validation_state", "validation_warning", "supported_claims", "unverified_claims"}:
            continue
        pieces.append(f"<h2>{escape(_display_label(key))}</h2>")
        if key == "documents_included" and isinstance(value, list):
            for item in value:
                pieces.append(f"<h3>{escape(str(item.get('document_name') or item.get('name') or 'Document'))}</h3>")
                if str(item.get("summary") or "").strip():
                    pieces.append(f"<p>{escape(str(item['summary'])).replace(chr(10), '<br>')}</p>")
                findings = item.get("findings", []) or []
                if findings:
                    pieces.append("<ul>" + "".join(f"<li>{escape(str(x))}</li>" for x in findings) + "</ul>")
        elif isinstance(value, list):
            pieces.append("<ul>" + "".join(f"<li>{escape(str(item))}</li>" for item in value) + "</ul>")
        elif isinstance(value, str) and value.strip():
            pieces.append(f"<p>{escape(value.strip()).replace(chr(10), '<br>')}</p>")
    html = "<html><body>" + "".join(pieces) + "</body></html>"
    css = """
      body { font-family: sans-serif; font-size: 10.5pt; color: #26342d; line-height: 1.4; }
      .brand { color: #27804f; font-size: 9pt; font-weight: bold; letter-spacing: 1.2pt; margin-bottom: 18pt; }
      h1 { color: #173c2b; font-size: 24pt; margin-bottom: 8pt; }
      h2 { color: #24764b; font-size: 14pt; margin-top: 17pt; margin-bottom: 6pt; }
      h3 { color: #315843; font-size: 11pt; margin-top: 10pt; }
      .meta { color: #58665e; font-size: 9pt; margin: 2pt 0; }
      .warning { color: #7a3b00; background: #fff1cf; border: 1px solid #e1ac44; padding: 9pt; }
      p { margin: 5pt 0 9pt 0; }
      li { margin-bottom: 4pt; }
    """
    page_rect = fitz.paper_rect("a4")
    content_rect = fitz.Rect(54, 54, page_rect.width - 54, page_rect.height - 54)
    output = io.BytesIO()
    writer = fitz.DocumentWriter(output)
    story = fitz.Story(html=html, user_css=css)
    more = True
    while more:
        device = writer.begin_page(page_rect)
        more, _filled = story.place(content_rect)
        story.draw(device)
        writer.end_page()
    writer.close()
    pdf = fitz.open(stream=output.getvalue(), filetype="pdf")
    total_pages = len(pdf)
    for index, page in enumerate(pdf, start=1):
        page.insert_text((54, page.rect.height - 28), f"Scribe  |  {index} / {total_pages}",
                         fontsize=8, fontname="helv", color=(0.34, 0.42, 0.37))
    return pdf.tobytes()
