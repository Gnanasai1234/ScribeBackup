# Data model (MVP)

Current response models are JSON dictionaries and Pydantic request models in `backend/app/main.py`.

## Document

`id`, `name`, `text` (server memory only), `language`, `status`, `created_at`, `pages[]`, `chunks[]`, `result`, `options`.

## Chunk

`id`, `text`, `page` (nullable), `section`, `language`. Page is populated when PDF extraction can map it.

## Summary result

`depth` (resolved value, retained for compatibility), `summary_depth_requested`, `summary_depth_resolved`, `purpose`, `audience`, `language`, `status`, `executive_summary`, `findings[]`, `claims[]`, `quality`, `stats` (including `context_chunks_used`), `limitations[]`.

## Claim and validation

`id`, `claim`, `status` (`SUPPORTED` or `UNAVAILABLE`), `sources[]` (`page`, `section`, `chunk_id`), `reason`.

## Production schema direction

Normalize the structures into User, Document, DocumentVersion, Chunk, Summary, Claim, ValidationResult, Evaluation, Audio, Feedback, ProcessingJob, and ProviderEvent. Keep binary data in object storage and preserve provider/model/prompt version and timestamps on generated records. Add tenant ownership and authorization before exposing the service to multiple users.
