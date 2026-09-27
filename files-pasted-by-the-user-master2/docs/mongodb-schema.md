# MongoDB schema

Implements the persistence layer described in `architecture/architecture.md`'s
"Production evolution" section. Models live in `backend/models/`, the
connection helper in `backend/database/mongodb.py`. No repository/ORM
layer on top of Motor -- services call collections directly.

Large binaries (PDF/DOCX source files, generated MP3s) are never stored
in MongoDB. Only a path or URL to them (`fileUrl`, `audioUrl`) is kept
here. If embeddings are added later, they belong in a vector DB, not
in the `chunks` collection.

## Collections

| Collection   | Model file                        | Purpose |
|--------------|------------------------------------|---------|
| `users`      | `models/user.py`                   | Basic account info |
| `documents`  | `models/document.py`               | Uploaded file metadata |
| `chunks`     | `models/chunk.py`                  | Document chunks for retrieval/RAG |
| `summaries`  | `models/summary.py`                | Generated summaries |
| `validation` | `models/validation.py`             | Groundedness check + per-claim results |
| `audio`      | `models/audio.py`                  | Generated narration metadata |
| `qa`         | `models/qa.py`                     | "Ask the Document" question/answer log |
| `feedback`   | `models/feedback.py`               | User feedback on a summary/audio/qa result |

## Relationships

```text
USER
 └── DOCUMENT
       ├── CHUNKS
       ├── SUMMARIES
       │      └── VALIDATION
       ├── AUDIO
       ├── QA
       └── FEEDBACK
```

References are plain ObjectId-as-string fields: `userId`, `documentId`,
`summaryId`.

## Indexes

Created by `backend/database/mongodb.py:create_indexes()`:

```text
users.email        (unique)
documents.userId
documents.createdAt
chunks.documentId
summaries.documentId
validation.summaryId
audio.documentId
qa.documentId
feedback.documentId
```

## What's not wired in yet

The FastAPI routes in `backend/app/main.py` still use the in-memory
`DOCS` dict described in `docs/schema.md`. These models and the Motor
connection are ready to use, but swapping `main.py` over to them
(and calling `create_indexes()` on startup) is a separate step.
`backend/services/document_service.py` shows the CRUD pattern to
follow for the rest of the collections.
