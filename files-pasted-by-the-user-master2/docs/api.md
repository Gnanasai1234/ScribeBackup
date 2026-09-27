# API reference

Base URL: `http://127.0.0.1:8000`. Interactive OpenAPI docs: `/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Runtime status for LLM, local OCR, embeddings, vector DB, retrieval mode, database, and TTS |
| POST | `/api/demo` | Load and analyze deterministic sample |
| POST | `/api/documents/upload` | Upload PDF, DOCX, TXT, JSON, JPG/JPEG, or PNG (`multipart/form-data`, field `file`; optional `ocr_language`: auto, en, hi, te, ta, kn) |
| GET | `/api/documents` | List current session's documents |
| GET | `/api/documents/{id}` | Document metadata and short text preview |
| POST | `/api/documents/{id}/process` | Analyze and validate a structured summary (`summary_depth`, `purpose`, `audience`, `language`) |
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

Errors use standard HTTP status codes and a user-safe `detail` string. Documents live in process memory and disappear when the API restarts. The processing route uses Sarvam AI as the primary LLM (`LLM_PROVIDER=sarvam`, `SARVAM_API_KEY`). Configured OpenRouter and Groq providers can serve as secondaries; local extractive processing is the final summary fallback. Provider credentials never reach the frontend. For `/audio`, pass `provider: "edge"`, a language-matched `voice_id`, and optional `rate` from `0.7` to `1.3` to generate MP3 audio. The online speech request contains only the validated summary text. `provider: "browser"` returns the script for a matching local system voice.

`summary_depth` accepts `quick`, `standard`, `deep`, `auto`, or `custom`. Auto considers chunk count and document complexity; results include `summary_depth_requested`, `summary_depth_resolved`, and `stats.context_chunks_used`. The older `depth` request key remains accepted for compatibility; new clients should send `summary_depth`.

For custom mode, supply `custom_detail` (`quick`, `standard`, or `deep`) and `custom_sections` (findings, facts, risks, decisions, recommendations, action_items). `purpose`, `audience`, and `language` are passed through analysis and summary generation. Cross-language output requires a configured LLM; translated claims are validated against original report chunks. GovReport references are retained only in the selected in-memory dataset document for evaluation; they are not sent into summarization/validation or written to MongoDB. Dataset listing defaults to `GOVREPORT_DEV_LIMIT`; evaluation subsets are controlled by `GOVREPORT_EVALUATION_LIMIT`. MongoDB and vector database settings are not currently connected to persistence or retrieval.

## OpenRouter configuration

Set `LLM_PROVIDER=sarvam` and `SARVAM_API_KEY` in `.env`; the default model is `sarvam-105b`. Set `OPENROUTER_API_KEY` or `GROQ_API_KEY` to enable a configured secondary provider. Requests use bounded chunk groups for map analysis, synthesis, claim validation, and up to two regeneration attempts. See the official [OpenRouter quickstart](https://openrouter.ai/docs/quickstart) and [chat completions reference](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request).
