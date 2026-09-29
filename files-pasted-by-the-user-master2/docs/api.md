# API reference

Base URL: `http://127.0.0.1:8000`. Interactive OpenAPI docs: `/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Runtime status for LLM, local OCR, embeddings, vector DB, retrieval mode, database, and TTS |
| POST | `/api/demo` | Load and analyze deterministic sample |
| POST | `/api/documents/upload` | Upload PDF, DOCX, TXT, CSV, JSON, JPG/JPEG, or PNG (`multipart/form-data`, field `file`; optional `ocr_language`: auto, en, hi, te, ta, kn) |
| POST | `/api/documents/upload-batch` | Upload a bounded list of documents (`files[]`); each document reports an independent upload status |
| GET | `/api/documents` | List current session's documents |
| GET | `/api/documents/{id}` | Document metadata and short text preview |
| GET | `/api/history?page=1&limit=20` | Current user's paginated document history; supports `search`, `file_type`, `status`, `language`, `date_from`, `date_to`, and `sort=newest|oldest` |
| GET | `/api/history?activity_type=combined_summary&limit=50` | Current user's activity events for the requested event type |
| POST | `/api/documents/{id}/process` | Analyze and validate a structured summary (`summary_depth`, `purpose`, `audience`, `language`) |
| POST | `/api/documents/process-batch` | Process isolated document IDs with bounded concurrency; returns a `COMBINED_SUMMARY` with document-labeled sections and original claim provenance |
| GET | `/api/documents/{id}/summary` | Validated summary result |
| GET | `/api/documents/{id}/validation` | Claim support checks |
| GET | `/api/documents/{id}/sources` | Extracted chunks and metadata |
| GET | `/api/dataset/govreport?split=train&limit=20` | Stream a bounded GovReport sample from Hugging Face |
| POST | `/api/dataset/govreport/{record_id}/load?split=train` | Load one sampled report into the current process |
| POST | `/api/documents/{id}/ask` | Lexical retrieval plus evidence-grounded answer (`question`) |
| POST | `/api/documents/compare` | Exact sentence diff (`file_a`, `file_b`) |
| GET | `/api/tts/languages` | Supported interface language codes |
| GET | `/api/tts/voices?language=te&provider=edge` | Neural voice options for the selected language |
| POST | `/api/documents/{id}/audio` | Play with a language-matched system voice or return neural MP3 from a validated summary |
| GET | `/api/evaluation/{id}` | Available and unavailable evaluation metrics |
| POST | `/api/evaluation/run` | Evaluate an analyzed GovReport document against its held-out reference summary |
| POST | `/api/evaluation/{id}/compare-depths` | Run Quick, Standard, and Deep on one loaded GovReport record |

Errors use standard HTTP status codes and a user-safe `detail` string. Document and analysis data persist in MySQL; each document query is scoped to the authenticated session owner. File and extracted-text SHA-256 hashes are indexed together with `user_id`, so equal content is a duplicate only for that same account. The processing route uses Sarvam AI as the primary LLM (`LLM_PROVIDER=sarvam`, `SARVAM_API_KEY`). Configured OpenRouter and Groq providers can serve as secondaries; local extractive processing is the final summary fallback. Provider credentials never reach the frontend. For `/audio`, pass `provider: "edge"`, a language-matched `voice_id`, and optional `rate` from `0.7` to `1.3` to generate MP3 audio. The online speech request contains only the validated summary text. `provider: "browser"` returns the script for a matching local system voice.

`summary_depth` accepts `quick`, `standard`, `deep`, `auto`, or `custom`. Auto considers chunk count and document complexity; results include `summary_depth_requested`, `summary_depth_resolved`, and `stats.context_chunks_used`. The older `depth` request key remains accepted for compatibility; new clients should send `summary_depth`.

For custom mode, supply `custom_detail` (`quick`, `standard`, or `deep`) and `custom_sections` (findings, facts, risks, decisions, recommendations, action_items). `purpose`, `audience`, and `language` are passed through analysis and summary generation. Cross-language output requires a configured LLM; translated claims are validated against original report chunks. GovReport references are retained only on the selected dataset document for evaluation; they are not sent into summarization or validation. Dataset listing defaults to `GOVREPORT_DEV_LIMIT`; evaluation subsets are controlled by `GOVREPORT_EVALUATION_LIMIT`. Document and workflow persistence uses MySQL as configured in `DATABASE_URL`.

## Guardrail limits

Limits are read from `.env` (invalid values fall back to the defaults shown here):

| Variable | Default | Applies to |
|---|---:|---|
| `MAX_UPLOAD_MB` | 25 | Each uploaded file |
| `MAX_TOTAL_UPLOAD_MB` | 80 | A batch or compare request |
| `MAX_DOCUMENTS_PER_BATCH` | 8 | Batch upload and processing |
| `MAX_CONCURRENT_DOCUMENTS` | 2 | Batch processing workers |
| `MAX_DOCUMENT_CHARS` | 1,000,000 | Extracted document text |
| `MAX_CHUNK_CHARS` | 4,000 | A source chunk (the pipeline default is 900) |
| `MAX_CONTEXT_CHARS` | 100,000 | Serialized model context |
| `MAX_OUTPUT_TOKENS` | 8,192 | Global output cap |
| `MAX_SUMMARY_TOKENS_LOW` | 3,000 | Quick/low-depth summary cap |
| `MAX_SUMMARY_TOKENS_MEDIUM` | 5,500 | Standard/medium-depth summary cap |
| `MAX_SUMMARY_TOKENS_HIGH` | 8,000 | Deep/high-depth summary cap |
| `MAX_LLM_RETRIES` | 1 | Provider and malformed JSON retry cap |
| `MAX_VALIDATION_RETRIES` | 2 | Claim validation attempts after the first |
| `SUMMARY_MAX_REGENERATION_ATTEMPTS` | 2 | Grounding-driven summary regeneration |
| `MAX_TRANSLATION_RETRIES` | 2 | Translation fallback calls after the primary |
| `MAX_PDF_PAGES` | 2,000 | Pages extracted from a PDF |
| `MAX_ARCHIVE_UNPACKED_MB` | 100 | Uncompressed DOCX archive size |
| `MIN_DOCUMENT_TEXT_CHARS` | 20 | Minimum meaningful extracted text |
| `MAX_QUESTION_CHARS` | 4,000 | Ask document question length |
| `MAX_SUMMARY_CHARS` | 40,000 | Final summary character cap |

Uploads are checked by extension and byte signature, then extraction is bounded before text reaches chunking or a model. Duplicate detection checks persistent file and extracted-text hashes for the authenticated owner. Batch combined summaries keep each independently generated summary and its claims under a document label; they do not ask a model to synthesize across all source documents in one prompt.

## OpenRouter configuration

Set `LLM_PROVIDER=sarvam` and `SARVAM_API_KEY` in `.env`; the default model is `sarvam-105b`. Set `OPENROUTER_API_KEY` or `GROQ_API_KEY` to enable a configured secondary provider. Requests use bounded chunk groups for map analysis, synthesis, claim validation, and up to two regeneration attempts. See the official [OpenRouter quickstart](https://openrouter.ai/docs/quickstart) and [chat completions reference](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request).
