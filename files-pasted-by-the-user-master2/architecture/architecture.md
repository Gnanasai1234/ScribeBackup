# Scribe architecture

```mermaid
flowchart TD
  U[Upload / demo] --> X[Text extraction and OCR]
  X --> C[Chunking and document analysis]
  C --> G[Shared LLM service]
  G --> O[OpenRouter primary]
  O -->|failure| A[API Gateway secondary /v1]
  A -->|failure| L[Ollama Local final fallback]
  G --> V[Claim validation and translation]
  V --> S[Persisted summary and actual provider metadata]
  Q[Question] --> R[Lexical / embedding retrieval]
  R --> E[Retrieved evidence]
  E --> G
  G --> Z[Grounded answer or controlled failure]
  R --> H[Embedding service: separate from generation]
  L -->|all generation providers fail| D[Summary: deterministic extractive fallback]
```

All general-purpose generation uses `backend/app/llm_service.py` and the strict OpenRouter ? API Gateway ? Ollama Local chain. The embedding subsystem is separate. Speech-to-text is a specialized service. Summary generation retains the deterministic extractive fallback after all three LLM providers fail, while RAG returns a controlled failure instead of presenting retrieved excerpts as an answer.
