# MySQL persistence

Scribe uses MySQL through SQLAlchemy. Set `DATABASE_URL` in the project root `.env`; the application does not fall back to another database.

```dotenv
DATABASE_URL=mysql+pymysql://root:YOUR_PASSWORD@localhost:3306/scribe?charset=utf8mb4
```

Create the database once, then apply the versioned schema:

```sql
CREATE DATABASE scribe CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

```powershell
cd backend
alembic upgrade head
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Tables store documents, pages, chunks, summaries, summary claims, validation results, translations, audio metadata, and Q&A history. Extracted text, summary, claim, translation, and narration text use MySQL `LONGTEXT`; document text is stored in relational document rows and chunks in individual rows. Files remain in the configured local storage system.

Migration `0009_summary_evaluations.py` adds owner-scoped evaluation rows keyed to a specific `summary_id` and evaluator version. Each row stores nullable metric scores and the complete JSON report. Reprocessing appends a summary and evaluation version instead of deleting the previous summary history. Apply it with the full migration chain before serving the new evaluation endpoint.

Migration `0002_users_and_document_ownership.py` adds users and a nullable `documents.user_id` foreign key. Existing rows are left with `user_id = NULL`; Scribe does not assign them to the first account. Assign only selected legacy rows after registering an account:

```sql
UPDATE documents
SET user_id = (SELECT id FROM users WHERE email = LOWER(TRIM('alice@example.com')))
WHERE id IN ('document-id-one', 'document-id-two')
  AND user_id IS NULL;
```

Migration `0005_user_scoped_document_hashes.py` adds nullable SHA-256 columns for uploaded file bytes and extracted text. Composite unique indexes on `(user_id, content_sha256)` and `(user_id, extracted_text_sha256)` prevent duplicate uploads within one account while allowing equal content in different accounts. Null hashes on older rows are retained; duplicate lookup hashes their extracted text lazily for the requesting owner. A `(user_id, created_at)` index supports paginated history. Apply the full migration chain with `alembic upgrade head` before starting the updated backend.

`SESSION_SECRET` must also be set in the root `.env` to a random string of at least 32 characters. Sessions are signed cookies; the cookie contains only the user ID. Application startup does not create tables or modify existing documents.
