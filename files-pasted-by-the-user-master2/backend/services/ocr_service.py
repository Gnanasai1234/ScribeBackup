"""Optional, local Tesseract OCR for scanned PDFs and standalone images."""
from __future__ import annotations

import io
import os
import shutil
from typing import Any


LANGUAGE_CODES = {"en": "eng", "hi": "hin", "te": "tel", "ta": "tam", "kn": "kan"}
_ENGINE_KEY: tuple[str, str | None] | None = None
_ENGINE_INFO: dict[str, Any] | None = None


def configure_tesseract(force: bool = False) -> dict[str, Any]:
    """Resolve and verify Tesseract once; TESSERACT_CMD takes precedence over PATH."""
    global _ENGINE_KEY, _ENGINE_INFO
    configured = os.getenv("TESSERACT_CMD", "").strip()
    path_candidate = shutil.which("tesseract")
    key = (configured, path_candidate)
    if not force and key == _ENGINE_KEY and _ENGINE_INFO is not None:
        return dict(_ENGINE_INFO)
    try:
        import pytesseract
    except ImportError:
        info = {"status": "python-package-missing", "message": "Install the Python package pytesseract to enable local OCR."}
        _ENGINE_KEY, _ENGINE_INFO = key, info
        return dict(info)
    executable = configured if configured and os.path.isfile(configured) else None
    invalid_config = bool(configured and not executable)
    if not executable and path_candidate and os.path.isfile(path_candidate):
        executable = path_candidate
    if not executable:
        reason = "TESSERACT_CMD points to a file that does not exist. " if invalid_config else ""
        info = {"status": "executable-not-found", "message": reason + "Tesseract OCR executable was not found; set TESSERACT_CMD to tesseract.exe or add it to PATH."}
        _ENGINE_KEY, _ENGINE_INFO = key, info
        return dict(info)
    pytesseract.pytesseract.tesseract_cmd = executable
    try:
        version = str(pytesseract.get_tesseract_version())
    except Exception as exc:
        info = {"status": "executable-start-failed", "message": f"Tesseract executable exists but could not start ({type(exc).__name__}). Check its permissions and installation."}
        _ENGINE_KEY, _ENGINE_INFO = key, info
        return dict(info)
    info = {"status": "ready", "version": version, "message": None}
    _ENGINE_KEY, _ENGINE_INFO = key, info
    return dict(info)


def _tesseract():
    info = configure_tesseract()
    if info["status"] != "ready":
        return None, info
    import pytesseract
    try:
        installed = set(pytesseract.get_languages(config=""))
    except Exception as exc:
        return None, {"status": "language-data-unavailable", "message": f"Tesseract started, but its language data could not be loaded ({type(exc).__name__})."}
    return (pytesseract, installed), {"status": "ready", "version": info.get("version"), "message": None}


def ocr_status() -> dict[str, Any]:
    api, info = _tesseract()
    return {"status": "local-ready" if api else info["status"], "languages": sorted(api[1]) if api else [], "version": info.get("version"), "message": info.get("message")}


def _language_config(pytesseract: Any, installed: set[str], language: str | None) -> tuple[str, str | None]:
    requested = LANGUAGE_CODES.get((language or "").lower(), "")
    if requested and requested not in installed:
        # Never use English recognition for an explicitly selected Indic script.
        return "", f"Required Tesseract language data '{requested}' is missing; OCR for this page was skipped."
    if requested:
        return requested, None
    # In auto mode, use only installed models. Script-based language detection is
    # run over extracted OCR output by the document pipeline after recognition.
    available = [code for code in ("eng", "hin", "tel", "tam", "kan") if code in installed]
    selected = "+".join(available or (["eng"] if "eng" in installed else []))
    return selected, None if selected else "Required Tesseract language data is missing; no supported OCR model is installed."


def _preprocess(image):
    """Conservative upscale, grayscale, contrast, and deskew fallback."""
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps

        rgb = image.convert("RGB")
        if min(rgb.size) < 1400:
            scale = min(3.0, max(1.5, 1400 / max(1, min(rgb.size))))
            rgb = rgb.resize((int(rgb.width * scale), int(rgb.height * scale)), Image.Resampling.LANCZOS)
        gray = ImageOps.grayscale(rgb)
        gray = ImageEnhance.Contrast(gray).enhance(1.35).filter(ImageFilter.MedianFilter(size=3))
        array = np.asarray(gray)
        binary = cv2.threshold(array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(binary < 250))
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords)[-1]
            angle = -(90 + angle) if angle < -45 else -angle
            if 0.5 < abs(angle) < 12:
                h, w = binary.shape
                matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                binary = cv2.warpAffine(binary, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(binary)
    except Exception:
        return image


def recognize_image(image, language: str | None = None) -> dict[str, Any]:
    api, diagnostic = _tesseract()
    if diagnostic["status"] != "ready":
        return {"text": "", "confidence": None, "status": diagnostic["status"], "warning": diagnostic["message"]}
    pytesseract, installed = api
    lang, language_warning = _language_config(pytesseract, installed, language)
    if not lang:
        return {"text": "", "confidence": None, "status": "missing-language-data", "warning": language_warning or "Required Tesseract language data is missing."}
    try:
        try:
            orientation = pytesseract.image_to_osd(image, config="--psm 0", output_type=pytesseract.Output.DICT)
            rotation = int(orientation.get("rotate", 0)) % 360
            if rotation in {90, 180, 270}:
                image = image.rotate(rotation, expand=True)
        except Exception:
            pass
        first = pytesseract.image_to_data(image, lang=lang, config="--psm 3", output_type=pytesseract.Output.DICT)
        confs = [float(c) for c, word in zip(first.get("conf", []), first.get("text", [])) if word.strip() and float(c) >= 0]
        text = pytesseract.image_to_string(image, lang=lang, config="--psm 3").strip()
        confidence = sum(confs) / len(confs) if confs else None
        if not text or (confidence is not None and confidence < 55):
            processed = _preprocess(image)
            retry = pytesseract.image_to_data(processed, lang=lang, config="--psm 11", output_type=pytesseract.Output.DICT)
            retry_confs = [float(c) for c, word in zip(retry.get("conf", []), retry.get("text", [])) if word.strip() and float(c) >= 0]
            retry_text = pytesseract.image_to_string(processed, lang=lang, config="--psm 11").strip()
            retry_conf = sum(retry_confs) / len(retry_confs) if retry_confs else None
            if len(retry_text) > len(text) or (retry_conf or 0) > (confidence or 0):
                text, confidence = retry_text, retry_conf
        warning = language_warning
        if confidence is not None and confidence < 40:
            warning = "Some image text was recognized with low confidence."
        if not text:
            try:
                from PIL import ImageStat
                visually_blank = ImageStat.Stat(image.convert("L")).stddev[0] < 1.5
            except Exception:
                visually_blank = False
            status = "image-no-readable-text" if visually_blank else "ocr-empty-result"
            warning = ("The image contains no visible text to recognize." if visually_blank else
                       "Tesseract ran successfully but returned empty text for an image with visible content.")
            return {"text": "", "confidence": round(confidence, 1) if confidence is not None else None, "status": status, "warning": warning}
        return {"text": text, "confidence": round(confidence, 1) if confidence is not None else None, "status": "success", "warning": warning}
    except Exception as exc:
        return {"text": "", "confidence": None, "status": "ocr-processing-failed", "warning": f"Tesseract executable was found, but OCR processing failed ({type(exc).__name__}); continuing with extracted text only."}


def extract_image_ocr(data: bytes, language: str | None = None) -> dict[str, Any]:
    try:
        from PIL import Image, ImageOps
        with Image.open(io.BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        result = recognize_image(image, language)
        result["records"] = [{"page": None, "source": "image", "text": result["text"], "ocr_confidence": result["confidence"]}] if result["text"] else []
        return result
    except Exception as exc:
        return {"text": "", "confidence": None, "records": [], "status": "image-load-failed", "warning": f"Image could not be opened for OCR ({type(exc).__name__}); continuing with extracted text only."}


def extract_pdf_ocr(data: bytes, pages: list[dict[str, Any]], language: str | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """OCR image/scanned pages and merge OCR text into their existing page records."""
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        import fitz
        from PIL import Image
    except ImportError:
        if any(len((p.get("text") or "").strip()) < 40 for p in pages):
            warnings.append("Install PyMuPDF and Pillow to scan image-based PDF pages; continuing with extractable text.")
        return pages, records, warnings
    try:
        pdf = fitz.open(stream=data, filetype="pdf")
        for index, page in enumerate(pdf):
            current = pages[index] if index < len(pages) else {"page": index + 1, "text": ""}
            text = (current.get("text") or "").strip()
            # Text-poor pages are scanned. Pages with embedded images are also
            # OCR'd so captions/paragraphs inside figures are retained.
            if len(text) >= 40 and not page.get_images(full=True):
                continue
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            result = recognize_image(image, language)
            if result.get("text"):
                record = {"page": index + 1, "source": "image", "text": result["text"], "ocr_confidence": result.get("confidence")}
                records.append(record)
                current["ocr"] = record
                current["text"] = (text + "\n\n[Image 1 OCR]\n" + result["text"]).strip()
            if result.get("warning"):
                warnings.append(result["warning"])
            if index >= len(pages):
                pages.append(current)
        pdf.close()
    except Exception as exc:
        warnings.append(f"PDF image scanning was unavailable ({type(exc).__name__}); continuing with extractable text.")
    return pages, records, list(dict.fromkeys(warnings))
