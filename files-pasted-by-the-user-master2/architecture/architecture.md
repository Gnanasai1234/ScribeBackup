# Architecture

```mermaid
flowchart LR
  U[React + TypeScript UI] --> A[FastAPI]
  A --> I[File validation and extraction]
  I --> P[Normalization and language heuristic]
  P --> C[Sentence-aware chunks with source metadata]
  C --> R[Lexical retrieval]
  C --> M[OpenRouter chunk analysis]
  M --> S[Structured synthesis]
  S --> V[Claim validation and bounded regeneration]
  V --> Q[Supported claims only feed validated summary]
  Q --> O[Validated summary]
  O --> T[Browser speech synthesis]
  R --> QE[Top evidence passages]
  QE --> QA[OpenRouter Q&A or source excerpt fallback]
```

The current build is a local demo vertical slice. It uses OpenRouter through the OpenAI-compatible chat completions interface, in-memory document metadata, and lexical retrieval. OCR, durable persistence, semantic embeddings, reranking, asynchronous job queues, and database migrations remain future implementation work and are not represented as complete.

## Data flow and safety invariants

1. Upload extension is checked, data is limited to 25 MiB, and extraction is format-specific.
2. Chunks retain a stable ID, detected language, section placeholder, and PDF page where available.
3. OpenRouter gets bounded chunk groups, then synthesizes a structured summary with chunk IDs. A validator checks each claim against only its cited passages. Up to two repair attempts are made.
4. If provider output fails or validation is unavailable, a deterministic extractive summary is used. Unsupported generated claims are not treated as validated.
5. Q&A retrieves lexical evidence first, then sends only those passages with the question to OpenRouter. When generation fails, the UI gets source excerpts.
6. The audio endpoint requires a completed `validated` summary and returns that summary text only.
7. When a capability is missing (OCR, semantic retrieval, translation, evaluation), the API reports it as unavailable instead of claiming success.

## Production evolution

Replace in-memory `DOCS` with PostgreSQL entities for documents, versions, chunks, processing jobs, summaries, claims, validations, audio, and feedback. Store file binaries outside the relational database. Move extraction and analysis into a queue worker. Introduce provider protocols and circuit breakers around model, embedding, OCR, and TTS calls. Add multilingual embeddings and reranking, then evaluate each component against a versioned benchmark before enabling it.
