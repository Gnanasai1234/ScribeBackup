# LLM configuration

All general-purpose text generation uses one backend-only provider router with a fixed order: **OpenRouter ? API Gateway ? Ollama Local**. Summarization, claim validation, translation, audio-script translation, document comparison, evidence-grounded answers, evaluation, and the LLM Provider Test all use this router. Frontend requests cannot select a provider or change the order.

## Configuration

Set these in the backend environment. Credentials are never logged or sent to the frontend.

| Variable | Default / purpose |
|---|---|
| `OPENROUTER_API_KEY` | OpenRouter credential |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` |
| `OPENROUTER_TIMEOUT_SECONDS` | 30 seconds, bounded to 5?90 |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` (custom values are rejected) |
| `OPENAI_API_KEY` | Direct OpenAI API credential; placeholders are rejected |
| `OPENAI_MODEL` | `gpt-4o-mini` |
| `OPENAI_TIMEOUT_SECONDS` | 30 seconds, bounded to 5?90 |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | `llama3.2:1b` |
| `OLLAMA_TIMEOUT_SECONDS` | 45 seconds, bounded to 5?90 |

General LLM generation uses one router in this order: OpenRouter, Direct OpenAI at `https://api.openai.com/v1`, then local Ollama. The OpenAI key is backend-only. Missing or placeholder keys are not sent; the service logs a safe configuration error and continues to Ollama. When every model provider fails, document summaries may use the existing deterministic extractive fallback. Ollama uses only an already-installed model and never pulls one automatically.

A provider succeeds only when it returns non-empty, valid output. The router records the provider, model, fallback reason, and providers actually attempted. Errors retain their original exception internally and log safe categories, status when available, and error types, without credentials, prompts, or response bodies.

When all providers fail, document summaries may use the existing deterministic extractive fallback, labeled `deterministic-extractive`. Evidence-grounded Q&A does not fabricate a response. Failed translation preserves the source text and reports the failure.

`POST /api/llm/diagnostics` is an authenticated, explicit generation probe. It independently tests OpenRouter with `OPENROUTER_TEST_OK`, Direct OpenAI with `OPENAI_DIRECT_TEST_OK`, and Ollama with `OLLAMA_TEST_OK`; normal startup performs no generation probes. `/api/health` reports safe configuration and local availability only. `POST /api/llm/test` uses the shared priority chain and returns the actual provider execution trace. `/api/local-llm/test` remains an authenticated compatibility alias.

Embeddings are a separate capability and may use their own configured provider. Specialized speech-to-text and text-to-speech services also remain separate from general-purpose LLM routing.
