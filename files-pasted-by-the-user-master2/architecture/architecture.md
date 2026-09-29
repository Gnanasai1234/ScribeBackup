# Scribe architecture

```mermaid
flowchart TD
  U[Upload / demo] --> X[Text extraction and OCR]
  X --> C[Chunking and document analysis]
  C --> G[Shared LLM service]
  G --> O[OpenAI Gateway primary]
  O -->|failure| M[Gemini fallback]
  M -->|failure| L[Ollama Local final fallback]
  G --> V[Claim validation and translation]
  V --> S[Persisted summary and actual provider metadata]
  Q[Question] --> R[Lexical / embedding retrieval]
  R --> E[Retrieved evidence]
  E --> G
  G --> A[Grounded answer or controlled failure]
  R --> H[Embedding service: Groq / Gateway / local]
```

All general-purpose generative operations use `backend/app/llm_service.py` and the strict OpenAI Gateway -> Gemini -> Ollama Local provider chain. The embedding subsystem is separate and may use its own configured services. Speech-to-text is a separate specialized service. Summary generation retains a deterministic extractive fallback after all three providers fail, while RAG does not expose retrieved excerpts as an answer when generation fails.
