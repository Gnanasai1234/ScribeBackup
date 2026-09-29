# Scribe — Document Intelligence MVP

Scribe is a document intelligence prototype that uses a centralized provider chain: Gemini, then an installed local Ollama model. Local extractive summarization and lexical evidence retrieval remain available as final fallbacks. Speech transcription can use hosted Whisper when configured and automatically falls back to local Faster Whisper.

## What works

- PDF, DOCX, TXT, CSV, JSON, JPG/JPEG, and PNG ingestion with a 25 MiB upload limit. Scanned PDF pages, pages containing embedded images, and standalone images are scanned with local Tesseract OCR when installed.
- Text normalization, sentence-aware overlapping chunks, PDF page retention, OCR confidence and page provenance, and conservative script-based language detection for English, Hindi, Telugu, Tamil, and Kannada.
- Quick, Standard, Deep, Auto, and Custom summary settings. The modes change context coverage, map-note detail, output structure, claim limits, and synthesis token budgets.
- Structured summary fields, claim-level validation against cited source chunks, and up to two regeneration attempts.
- Evidence retrieval followed by an LLM-generated Q&A response constrained to retrieved source passages, using the centralized provider chain.
- Deterministic extractive summaries remain available when Gemini and Ollama are unavailable.
- Exact sentence additions/removals for document comparison.
- Telugu, Hindi, and Tamil translation through the configured LLM when requested. Translated claims are checked against the original-language report again.
- Language-matched speech from validated summaries: local system voices when available, with downloadable online neural MP3 playback for English, Hindi, Telugu, and Tamil.
- A Local LLM Test workspace for text and voice prompts. Voice uses offline Faster Whisper transcription and sends the reviewed transcript through the configured provider chain.
- Deterministic demo data and OpenAPI documentation.

## Current limits

This is a runnable MVP, not the entire requested production platform. The general-purpose LLM chain is OpenAI Gateway -> Gemini -> Ollama Local. Scribe has email/password accounts and signed cookie sessions, with each document assigned to its uploader. Embeddings, a vector database, a background queue, and persistent generated audio files are not connected. Document and analysis persistence uses MySQL configured through `DATABASE_URL`; see [docs/database.md](docs/database.md). OCR uses free local Tesseract and continues with any normally extracted text when OCR is unavailable. Speech can use local system voices or the optional Edge neural TTS service; online voice generation sends only the validated summary text to Microsoft for synthesis. Claim validation uses the same centralized provider service, so treat it as an automated check rather than independent human verification.

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

Before starting the backend, set `SESSION_SECRET` to a random value of at least 32 characters in `.env`, and configure your existing `DATABASE_URL`. Generate a secret in PowerShell with `python -c "import secrets; print(secrets.token_urlsafe(48))"`. Hosted LLM and Whisper API keys are optional. Set `OPENAI_GATEWAY_URL` and `OPENAI_API_KEY` for the primary provider, `GEMINI_API_KEY` / `GEMINI_MODEL` for the secondary provider, and an already-installed model selected by `OLLAMA_MODEL` (defaults to `llama3.2:1b`) at `OLLAMA_BASE_URL` (defaults to `http://127.0.0.1:11434`). Make sure Ollama is running and the model has already been pulled. Speech transcription uses the already-cached Faster Whisper model selected by `LOCAL_STT_MODEL` (defaults to `base`); it does not download a model during a request. Provider selection and fallback are logged by the backend.

Apply database changes from the backend directory with `alembic upgrade head`. The migrations add `users`, a nullable `documents.user_id`, and per-user upload/text hash indexes for duplicate detection; existing documents stay unassigned until you choose them. To assign selected legacy rows to an account, first create that account through Register, then run a deliberate MySQL update, replacing the email and IDs with your choices:

```sql
UPDATE documents
SET user_id = (SELECT id FROM users WHERE email = LOWER(TRIM('alice@example.com')))
WHERE id IN ('document-id-one', 'document-id-two')
  AND user_id IS NULL;
```

```powershell
Copy-Item .env.example .env
notepad .env
```

Document text and retrieved passages are sent first to the configured Gemini provider, then to Ollama only when Gemini fails. The provider that actually generated the response is recorded in the result. Review provider data handling settings before processing sensitive files.

Open `http://localhost:5173` so the frontend and API use the same site for the session cookie. API docs are at `http://localhost:8000/docs`. The demo button loads a fictional transit review and processes it through the configured OpenAI Gateway -> Gemini -> Ollama generation flow.


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

Keep `.env` on the backend and never place provider keys in frontend configuration. Configure `OPENAI_GATEWAY_URL`, `OPENAI_API_KEY`, and optionally `OPENAI_MODEL`; configure `GEMINI_API_KEY`, `GEMINI_MODEL`, and `GEMINI_TIMEOUT_SECONDS`; configure `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and `OLLAMA_TIMEOUT_SECONDS`. Groq remains limited to embeddings. `SUMMARY_MAX_REGENERATION_ATTEMPTS` sets the retry limit. `EMBEDDING_PROVIDER` and `VECTOR_DB_URL` are currently configuration placeholders and do not activate vector retrieval.

Voice prompts use `LOCAL_STT_MODEL` (defaults to `base`) through Faster Whisper on the backend CPU. The selected model must already be present in the local Hugging Face cache; transcription runs with offline-only model loading and does not call a paid speech API. `LOCAL_STT_DEVICE` and `LOCAL_STT_COMPUTE_TYPE` can override the default `cpu` and `int8` settings.

## API and schemas

See [API reference](docs/api.md), [LLM setup and request flow](docs/llm.md), [schema notes](docs/schema.md), and the generated OpenAPI UI. The architecture diagram and trust boundaries are in [architecture/architecture.md](architecture/architecture.md).

## Evaluation and tests

See [evaluation plan](evaluation/README.md). Run backend checks from the repository root with:

```powershell
python -m unittest discover -s backend/tests -v
```

Build the frontend with `cd frontend; npm run build`.

## Next production steps

Connect the configured vector service, add persistent audio file storage, move processing to a job worker, and add observability. Semantic similarity and several reference-independent quality metrics are currently reported as unavailable rather than estimated.
