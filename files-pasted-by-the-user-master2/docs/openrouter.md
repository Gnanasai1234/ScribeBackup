# OpenRouter integration

Scribe uses the OpenAI Python SDK pointed at OpenRouter's compatible chat-completions endpoint. The OpenRouter key stays in the FastAPI process; it is never returned to React.

## Configure

1. Create an OpenRouter API key.
2. Copy `.env.example` to `.env` at the repository root.
3. Set `OPENROUTER_API_KEY` and choose a model slug in `OPENROUTER_MODEL`.
4. Install backend dependencies and start FastAPI as described in the root README.

The default slug is `openai/gpt-4o-mini`. OpenRouter currently documents this model slug and its chat-completions use at [the model page](https://openrouter.ai/openai/gpt-4o-mini/overview). Choose any model available to your account; model features, limits, latency, and usage charges vary.

## Request flow

- **Analysis:** chunks are sent in bounded groups of eight. Quick ranks and maps about one third of the source chunks, uses compact notes, a concise synthesis instruction, and a 1,400-token synthesis limit. Standard maps all chunks with balanced notes and a 2,600-token limit. Deep maps all chunks with expanded note detail and a 4,400-token limit. Claim limits and the final validated executive summary coverage also vary by mode. Auto resolves from chunk count and document complexity. Custom uses its selected detail profile and only requests selected output sections.
- **Purpose and audience:** the chosen purpose changes the focus/instructions (for example, risk analysis emphasizes risks); the audience setting changes the writing style and expected level of detail. Both are passed through the request and workflow state.
- **Validation:** up to eight claims and their cited evidence are sent per validation request. Returned source IDs are restricted to the IDs each claim cited. Unsupported or partial claims trigger a bounded rewrite and another validation pass, up to two times.
- **Language:** generation begins in the source language. When a different supported output language is requested, the generated summary is translated, then its claims are checked again against the original report chunks. A translation is not treated as grounded solely because its source-language version passed.
- **Audio gate:** only a `validated` summary reaches the audio endpoint. Matching local voices use browser synthesis; the optional `edge-tts` path creates MP3 audio with language-matched neural voices and exposes browser playback/download controls. Online synthesis sends only the validated summary text to Microsoft Edge's online speech service.
- **Q&A:** lexical retrieval selects up to three evidence chunks first. Only those passages and the question are sent to OpenRouter. If the call fails, Scribe returns the passages without a generated answer.
- **Fallback:** if the key is missing or an analysis request fails, Scribe creates a deterministic extractive summary with exact source-chunk checks and marks the fallback in the result.
- **Auto:** resolves by extracted chunk count: up to 8 chunks is quick, 9–30 is standard, and more than 30 is deep. The response reports both requested and resolved values.

The structured request fields, resolved depth, purpose, audience, and output language are retained in the result. With no configured key, the extractive fallback is deterministic and does not infer purpose-specific risks or translate text; the response includes the actual generated language and limitations.

The implementation lives in `backend/app/openrouter.py`; route integration and fallback behavior live in `backend/app/main.py`.

## Model calls and privacy

The number of map calls grows with document chunk count; synthesis, validation, and any regeneration add more calls. Requests can incur OpenRouter/provider charges. Extracted document content is sent to the selected model provider for processing, so review applicable data-handling settings before uploading sensitive documents.

## Official references

- [OpenRouter quickstart](https://openrouter.ai/docs/quickstart)
- [Chat completions API](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request)
- [OpenRouter GPT-4o mini model details](https://openrouter.ai/openai/gpt-4o-mini/overview)
