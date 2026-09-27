# Scribe — Document Intelligence MVP

Scribe is a document intelligence prototype that uses Sarvam AI as its primary LLM for chunk-based analysis, structured summaries, claim validation, bounded regeneration, and evidence-grounded Q&A. OpenRouter and Groq can serve as configured secondary providers. Local extractive summarization and lexical evidence retrieval remain available as fallbacks.

## What works

- PDF, DOCX, TXT, JSON, JPG/JPEG, and PNG ingestion with a 25 MiB upload limit. Scanned PDF pages, pages containing embedded images, and standalone images are scanned with local Tesseract OCR when installed.
- Text normalization, sentence-aware overlapping chunks, PDF page retention, OCR confidence and page provenance, and conservative script-based language detection for English, Hindi, Telugu, Tamil, and Kannada.
- Quick, Standard, Deep, Auto, and Custom summary settings. The modes change context coverage, map-note detail, output structure, claim limits, and synthesis token budgets.
- Structured summary fields, claim-level validation against cited source chunks, and up to two regeneration attempts.
- Evidence retrieval followed by a configured Sarvam AI Q&A response constrained to retrieved source passages, with configured secondary and lexical excerpt fallbacks.
- Deterministic extractive mode remains available when the configured LLM and secondary providers are unavailable.
- Exact sentence additions/removals for document comparison.
- Telugu, Hindi, and Tamil translation through the configured LLM when requested. Translated claims are checked against the original-language report again.
- Language-matched speech from validated summaries: local system voices when available, with downloadable online neural MP3 playback for English, Hindi, Telugu, and Tamil.
- Bounded GovReport streaming, in-memory report selection, reference-only ROUGE evaluation, and same-document Quick/Standard/Deep comparison.
- Deterministic demo data and OpenAPI documentation.

## Current limits

This is a runnable MVP, not the entire requested production platform. Sarvam AI is the primary generation provider when its API key is configured; OpenRouter and Groq are secondary providers. Embeddings, a vector database, MongoDB persistence, a background queue, authentication, and persistent audio history are not connected. OCR uses free local Tesseract and continues with any normally extracted text when OCR is unavailable. Speech can use local system voices or the optional Edge neural TTS service; online voice generation sends only the validated summary text to Microsoft for synthesis. GovReport records and references are held in process memory only; the reference summary is used only by evaluation. Claim validation is another model call using the same selected provider, so treat it as an automated check rather than independent human verification. Documents disappear when the API restarts.

## Requirements

- Python 3.10+
- Node.js 18+

## Run locally

In one terminal:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Before starting the backend, set `LLM_PROVIDER=sarvam` and `SARVAM_API_KEY` in `.env`. The default primary model is `sarvam-105b` at `https://api.sarvam.ai/v1`. OpenRouter (`OPENROUTER_API_KEY`) and Groq (`GROQ_API_KEY`) remain available as secondary providers. The configured provider can be selected with `LLM_PROVIDER`; fallback to local summarization remains available.

```powershell
Copy-Item .env.example .env
notepad .env
```

With Sarvam configured, extracted document chunks are sent to Sarvam AI for analysis, validation, and Q&A. Provider requests can incur usage charges. Check the selected model and provider data handling settings before processing sensitive files.

Open `http://127.0.0.1:5173`. API docs are at `http://127.0.0.1:8000/docs`. The demo button loads a fictional transit review using the deterministic local pipeline without calling OpenRouter.


## Local OCR setup (Windows)

OCR runs on the backend computer and does not need an OCR API key. Python dependencies (pytesseract, Pillow, OpenCV, and PyMuPDF) are installed by `pip install -r requirements.txt`; Tesseract itself is a separate Windows program.

1. Install the free Tesseract OCR Windows build from [UB Mannheim's installer page](https://github.com/UB-Mannheim/tesseract/wiki). In the installer, enable English and any additional language data you need. Alternatively, install Tesseract and download language files using step 5.
2. Add the Tesseract install directory to Windows PATH: open **Start ? Edit the system environment variables ? Environment Variables**, select **Path** under your user variables, choose **Edit ? New**, add the directory containing `tesseract.exe`, and confirm. Open a new PowerShell window afterwards. You can skip PATH and set `TESSERACT_CMD` in the project `.env` to the executable path instead.
3. Verify the install in a new PowerShell terminal:

   ```powershell
   tesseract --version
   tesseract --list-langs
   ```

   The language list should include `eng` and the language codes you plan to OCR (`hin`, `tel`, `tam`, `kan`).

   Verify the executable selected by the backend from the `backend` directory with the activated virtual environment:

   ```powershell
   python -c "from dotenv import load_dotenv; load_dotenv('../.env'); from services.ocr_service import configure_tesseract; d=configure_tesseract(force=True); import pytesseract; print('Diagnostic:', d); print('Tesseract:', pytesseract.get_tesseract_version() if d['status']=='ready' else d['message']); print('Executable:', pytesseract.pytesseract.tesseract_cmd)"
   ```
4. Install the Python packages from the repository root (or from `backend`):

   ```powershell
   cd backend
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

5. To add or update language data, derive the `tessdata` directory from the executable discovered on PATH, then download the free Tesseract community models. Run in PowerShell:

   ```powershell
   $tessdata = Join-Path (Split-Path (Get-Command tesseract).Source) 'tessdata'
   foreach ($lang in 'eng','hin','tel','tam','kan') {
     Invoke-WebRequest "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/$lang.traineddata" -OutFile (Join-Path $tessdata "$lang.traineddata")
   }
   tesseract --list-langs
   ```

   These are free, open-source language models. The upload screen's **OCR input language** selector lets you choose Auto, English, Hindi, Telugu, Tamil, or Kannada. Auto uses the installed models; when a language is explicitly selected but its model is missing, OCR reports that missing model rather than interpreting the page as English.
6. If the backend cannot find Tesseract, reopen the terminal after changing PATH and check `Get-Command tesseract`. Add `TESSERACT_CMD=` to the root `.env` to use PATH lookup. If PATH lookup is unsuitable, set `TESSERACT_CMD=C:\path\to\tesseract.exe` there (replace with the path shown on your computer) and restart the backend. If OCR is unavailable, the app continues with extracted PDF text and reports a non-blocking warning.

OCR text is merged into the document's normal page text before chunking, summarization, translation, and source validation. The original OCR text and confidence are retained with page/image provenance. Very low confidence is surfaced as a warning; OCR does not guess at unclear characters.

## Blank page troubleshooting

- Start the frontend from the `frontend` directory and leave that terminal running: `npm run dev`.
- If the browser only shows “Starting Scribe”, inspect the Vite terminal for a missing dependency or compile error.
- If React throws during rendering, the page now displays the error message instead of remaining blank. A WebGL/Three.js failure falls back to a text diagram.
- The landing screen can render without the backend, but upload, demo, Q&A, and analysis require FastAPI running on port 8000.
- Hard-refresh the page after correcting the Vite error.

## Environment

See [.env.example](.env.example). Keep `.env` on the backend and never place the OpenRouter key in frontend configuration. Without a Sarvam key, the configured OpenRouter or Groq secondary provider is used when available; otherwise, uploaded documents use the local extractive fallback. `GOVREPORT_DATASET_ID`, `GOVREPORT_SPLIT`, `GOVREPORT_DEV_LIMIT`, `GOVREPORT_EVALUATION_LIMIT`, and optional `GOVREPORT_CACHE_DIR` configure bounded dataset streaming. `SUMMARY_MAX_REGENERATION_ATTEMPTS` sets the retry limit. `EMBEDDING_PROVIDER` and `VECTOR_DB_URL` are currently configuration placeholders and do not activate vector retrieval.

## API and schemas

See [API reference](docs/api.md), [OpenRouter setup and request flow](docs/openrouter.md), [schema notes](docs/schema.md), and the generated OpenAPI UI. The architecture diagram and trust boundaries are in [architecture/architecture.md](architecture/architecture.md).

## Evaluation and tests

See [evaluation plan](evaluation/README.md). Run backend checks from the repository root with:

```powershell
python -m unittest discover -s backend/tests -v
```

Build the frontend with `cd frontend; npm run build`.

## Next production steps

Connect MongoDB and the configured vector service, add OCR and persistent audio history, move processing to a job worker, and add access control and observability. Semantic similarity and several reference-independent quality metrics are currently reported as unavailable rather than estimated.
