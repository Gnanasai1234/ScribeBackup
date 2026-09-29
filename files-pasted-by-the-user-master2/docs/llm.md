# LLM configuration

All general-purpose text generation uses one backend-only provider router. The strict order is **OpenAI Gateway → Gemini → Ollama Local** for summarization, claim validation, translation, audio-script translation, document comparison, and evidence-grounded answers. Frontend requests cannot select providers.

## Configuration

Set these in the backend environment. Keys are never logged or sent to the frontend.

| Variable | Default / purpose |
|---|---|
| `OPENAI_GATEWAY_URL` | `https://keygateway1.arshnivlabs.com/v1` (OpenAI-compatible base URL) |
| `OPENAI_API_KEY` | Gateway credential; `OPENAI_GATEWAY_API_KEY` remains a compatibility alias |
| `OPENAI_MODEL` | `gpt-4o-mini` |
| `OPENAI_TIMEOUT_SECONDS` | 30 seconds, bounded to 5–90 |
| `GEMINI_API_KEY` | Gemini Developer API credential |
| `GEMINI_MODEL` | `gemini-3.8-flash` |
| `GEMINI_TIMEOUT_SECONDS` | 45 seconds, bounded to 5–90 |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | `llama3.2:1b` |
| `OLLAMA_TIMEOUT_SECONDS` | 45 seconds, bounded to 5–90 |

The Gateway uses the OpenAI SDK pointed at its configured compatible URL. Gemini uses Google's `google-genai` SDK (`google-genai>=1.0,<2`), initialized with `GEMINI_API_KEY`; `GEMINI_TIMEOUT_SECONDS` controls its SDK HTTP timeout independently of Ollama. The model name is read only from `GEMINI_MODEL`. Ollama uses only already-installed models and never pulls a model.

The router attempts the Gateway first, then Gemini, then Ollama. A provider is skipped only after a previous provider returns usable output. Empty, blocked, malformed, or invalid output counts as failure. Metadata identifies the actual successful provider, model, and fallback reason. Failures log provider, model, status when available, categorized exception type, safe error details, and timeout, without credentials, request contents, or response bodies.

When all three providers fail, document summaries may use the deterministic extractive fallback, explicitly tagged `deterministic-extractive`. Evidence-grounded Q&A does not fabricate a response and returns its controlled failure. Translation preserves original-language content and exposes a warning when no provider can produce a valid translation.

`POST /api/llm/diagnostics` is an authenticated, explicit diagnostic that probes each provider separately. It reports safe key presence, request state, response state, and categorized errors, never credentials. It is not run at startup. `/api/health` reports non-generative configuration and local availability only. `POST /api/llm/test` uses the same priority chain and returns the provider execution trace. `/api/local-llm/test` remains an authenticated compatibility alias.

Embedding providers are configured separately and are not generative LLMs. Specialized speech-to-text and text-to-speech services also remain separate from general-purpose LLM routing.
