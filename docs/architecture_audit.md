# Architecture Audit

Date: 2026-09-07

Scope: repository inspection only. This audit does not add a local LLM, alter
RAG or Streamlit behavior, change the Firebase schema, or add training code.

## Executive Summary

The repository has two application entry paths:

1. `streamlit_app.py` is the documented interactive chatbot. It owns the
   chatbot UI, response branching, Firebase logging, ratings, and internal
   review screen.
2. `main.py` starts a FastAPI application from `src/api/endpoint.py`. The
   static browser client calls this API, but this path currently has no
   Firebase logging, rating, or review integration.

Both paths share `ContextRetriever`, `ReasoningChain`, `build_prompt`, and
`LLMFactory`. The LLM interface is OpenAI-compatible and asynchronous, so a
local OpenAI-compatible inference server is a reasonable future integration
target. The local provider branch is incomplete: it references a settings
attribute that does not exist, and both application paths select the Gemini
model name for every provider other than OpenAI.

There is also an observed data/runtime mismatch. The checked-in processed
chunks contain `metadata` fields such as `filename`, `document_number`, and
`category`, but no `intent_code`, `required_entities`, or
`clarifying_question`. `ReasoningChain` therefore receives `intent=None` and
normally returns clarification after retrieval. The LLM generation branch is
implemented, but is not reachable for those normal requests unless the
processed data has the fields expected by `ReasoningChain`.

## Current Application Flow

### Streamlit runtime

The documented flow is:

```text
User
  -> streamlit_app.py
  -> emergency keyword branch (some requests bypass RAG)
  -> ContextRetriever.get_context()
  -> retrieved contexts and entities
  -> ReasoningChain.process()
  -> build_prompt(context, query)
  -> LLMFactory.get_llm()
  -> AsyncOpenAI-compatible LLM provider
  -> streamed response
  -> Firebase chat log
  -> user rating / internal review
```

The exact Streamlit sequence is:

1. The module initializes Streamlit session state for messages, a session ID,
   and per-session submitted feedback.
2. `load_rag_components_v2()` is cached with `st.cache_resource`. It creates a
   `ContextRetriever`, a `ReasoningChain`, an LLM client, and loads all JSONL
   records from `settings.CHUNKS_FILE` for reasoning.
3. A query is checked against hard-coded information and emergency keywords.
   An emergency response is generated immediately, logged as `emergency`, and
   does not call the retriever or LLM.
4. Other queries call `ContextRetriever.get_context(prompt)`. If no contexts
   are returned, a fixed `no_data` response is logged and the LLM is skipped.
5. The first context's `metadata.intent_code` is passed to
   `ReasoningChain.process()`. If it is not ready, a clarification response is
   logged as `clarification` and the LLM is skipped.
6. For a ready request, retrieved text is joined with `---` separators and
   repeated periods are removed. `build_prompt()` inserts that text and the
   query into the system prompt.
7. Previous Streamlit messages are sent as chat history, followed by the
   current prompt. The provider is called with streaming enabled,
   `temperature=0.3`, and `max_tokens=2048`.
8. The complete response is appended to session state, logged as `normal`,
   and shown with feedback controls. LLM exceptions are shown and logged as
   `error`.

Relevant implementation: `streamlit_app.py:12-44`, `streamlit_app.py:151-331`.

### FastAPI/static runtime

The alternate flow is:

```text
Browser static/index.html + static/script.js
  -> POST /chat
  -> emergency branch OR ContextRetriever
  -> no-data branch OR ReasoningChain
  -> clarification branch OR build_prompt
  -> LLMFactory client
  -> Server-Sent Events response
```

`src/api/endpoint.py` creates the retriever, reasoning chain, LLM client, and
chunks at module import time. It streams metadata, response deltas, and a
`[DONE]` marker. It sends only the current prompt to the LLM; unlike
Streamlit, it does not send chat history. It does not call
`collect_chat_log`, `update_chat_feedback`, or any review function.

Relevant implementation: `main.py`, `src/api/endpoint.py:18-135`,
`static/script.js:166-220`.

## Current RAG Flow

### Offline ingestion/index construction

```text
data/raw/**/*.{pdf,docx,txt}
  -> ingestion_pipeline.py
  -> document_loaders.load_all_documents()
  -> custom_chunker.chunk_text()
  -> metadata_extractor.extract_metadata_from_file()
  -> VectorIndexer.prepare_chunks()
  -> chunks.jsonl and FAISS index/metadata.json
```

`VectorIndexer` embeds passages with `settings.EMBEDDING_MODEL`, prefixes E5
documents with `passage:`, builds a normalized inner-product FAISS index, and
saves the index and metadata. The current index is already present under
`data/processed/vector_index`; `chunks.jsonl` is loaded separately for the
reasoning step.

### Online retrieval

```text
query
  -> QueryAnalyzer.extract_entities()
  -> Retriever.retrieve()
  -> SentenceTransformer query embedding
  -> FAISS top-k search
  -> similarity threshold filter
  -> Grader.is_sufficient()
  -> {status, contexts, entities}
```

`Retriever` loads the embedding model, FAISS index, and `metadata.json`. E5
queries receive the `query:` prefix. It searches `TOP_K` records and keeps
scores at or above `SIMILARITY_THRESHOLD`. `ContextRetriever` always returns
the retrieved list and extracted entities, with status `success` or
`insufficient`; the Streamlit and API callers primarily use whether
`contexts` is empty.

The retrieval-related modules are:

- `src/rag_flow/retriever.py`: embedding model, FAISS loading/search, score
  filtering.
- `src/rag_flow/context_retriever.py`: orchestration of analyzer, retriever,
  and grader.
- `src/rag_flow/query_analyzer.py`: Vietnamese normalization helper and
  heuristic extraction of case IDs, phone numbers, and address snippets.
- `src/rag_flow/grader.py`: heuristic context sufficiency check.
- `src/rag_flow/reasoning_chain.py`: required-entity/clarification gate before
  generation.
- `src/ingestion/document_loaders.py`: PDF/DOCX/TXT loading.
- `src/ingestion/custom_chunker.py`: paragraph/sentence chunking.
- `src/ingestion/metadata_extractor.py`: source and document metadata.
- `src/ingestion/indexer.py`: embedding and FAISS index generation.

## Prompt Construction

`src/llm/prompt_templates.py` contains one `SYSTEM_PROMPT` and
`build_prompt(context, query)`. The template combines a long Vietnamese
assistant policy, the retrieved context, and the user query into one string.
The Streamlit path sends this as the current user message after prior chat
history. The API path sends the prompt as its only message.

Prompt construction is independent of the provider protocol, but it is a
shared behavioral contract: changes here affect Streamlit, FastAPI, evaluation,
and any future fine-tuning format that attempts to reproduce runtime behavior.

## Current LLM Architecture

### Factory and providers

`src/llm/model_factory.py` exposes `LLMFactory.get_llm()` and returns an
`openai.AsyncOpenAI` client:

| Provider value | Client initialization | Current model selection |
| --- | --- | --- |
| `openai` | `AsyncOpenAI(api_key=settings.OPENAI_API_KEY)` | `settings.OPENAI_MODEL` |
| `gemini` | `AsyncOpenAI(api_key=settings.GEMINI_API_KEY, base_url=Google OpenAI-compatible URL)` | `settings.GEMINI_MODEL` |
| `local` | Intended to use `AsyncOpenAI(base_url=settings.LOCAL_LLM_URL, api_key="dummy")` | Incorrectly uses `settings.GEMINI_MODEL` in both callers |

Unsupported values raise `ValueError`. There is no provider interface beyond
the common OpenAI-compatible client shape, no separate provider module, and no
model-name method in the factory. The factory is called once per application
instance: Streamlit caches it, while FastAPI creates it at module import.

### `LLM_PROVIDER` handling

`configs/settings.py` reads `LLM_PROVIDER` directly with default `gemini`.
Values are not trimmed, lowercased, or validated in settings. The factory
compares exact lowercase strings. The current local `.env` selects `gemini`.

The provider decision is duplicated in three places:

- `src/llm/model_factory.py` decides which client/base URL to create.
- `streamlit_app.py` chooses the model name for a request and for log metadata.
- `src/api/endpoint.py` chooses the model name for a request.

This duplication means provider integration changes must be coordinated across
the factory and both call paths.

## Firebase Logging, Feedback, and Review

### Implemented data flow

```text
Response branch
  -> log_response() in streamlit_app.py
  -> collect_chat_log()
  -> build_chat_record()
  -> save_chat_log()
  -> Firestore collection chat_logs
  -> user rating update OR admin review update
  -> dataset exporter
```

All Streamlit response branches call `log_response()`:

- `emergency`: fixed emergency text, empty contexts/entities.
- `no_data`: fixed no-context response.
- `clarification`: reasoning clarification plus contexts/entities.
- `normal`: generated answer plus contexts/entities.
- `error`: exception text plus contexts/entities.

`build_chat_record()` creates or preserves `session_id` and `turn_id`, adds an
UTC timestamp, query, answer, response type, normalized contexts, entities,
model, provider, and these review fields:

```text
review_status = "raw"
feedback = None
corrected_answer = None
```

Each context is reduced to `text`, `metadata`, and `score` before persistence.

`src/training_data/firebase_store.py` initializes Firebase lazily and caches
the Firestore client under a lock. Logging is best-effort: disabled logging,
missing credentials, initialization failures, and write failures return
`None`/`False` or an empty list and do not intentionally stop chat. The
collection name defaults to `chat_logs` and is configurable.

### User rating

After an assistant message, Streamlit offers `up`, `down`, or `wrong` ratings.
The rating is written under the nested `feedback` field with a UTC
`feedback_at` timestamp. The UI prevents repeated submission for the same turn
within the current Streamlit session. The update locates the first Firestore
document whose `turn_id` matches.

### Internal review

The Streamlit sidebar has a `Review data` view. It lists up to 100 Firestore
records and can filter by `review_status` or `feedback.rating`. An admin can
enter `corrected_answer`, set `error_type`, and choose one of:

```text
raw | approved | rejected | edited
```

The update writes the review fields back to the same document. There is no
separate review collection or audit history in the current schema.

Relevant modules: `src/training_data/collector.py`,
`src/training_data/firebase_store.py`, and `streamlit_app.py:54-141` plus
`streamlit_app.py:163-205`.

## Dataset Export and Training Boundary

Current locations and responsibilities:

- `src/training_data/anonymizer.py`: recursive redaction for email, Vietnamese
  phone, possible identity numbers, and case/file IDs.
- `scripts/export_sft_dataset.py`: fetches logs, keeps `approved` and `edited`,
  uses `corrected_answer` for edited records when non-empty, redacts records,
  and writes OpenAI-style `messages` JSONL.
- `scripts/export_preference_dataset.py`: fetches logs with a non-empty
  `corrected_answer`, redacts them, and writes `prompt`, `chosen`, `rejected`,
  and `reason` JSONL.
- `scripts/validate_training_exports.py`: validates JSONL structure and warns
  on common raw PII patterns.
- `data/training/exports/sft_dataset.jsonl`: existing SFT output artifact.
- `data/training/exports/preference_dataset.jsonl`: expected output path; the
  file is currently absent.
- `data/eval/eval_questions.jsonl`: evaluation-question artifact.

The effective current flow is:

```text
Firestore chat_logs
  -> list_chat_logs()
  -> anonymizer
  -> SFT or preference builder
  -> JSONL under data/training/exports
  -> validation
  -> future external training job
```

Important export semantics:

- SFT excludes `raw` and `rejected` records.
- Preference export currently checks only for a corrected answer. It does not
  require `review_status=edited` or `approved`, so raw or rejected records
  could be exported if they have a corrected answer.
- User ratings are not used directly by either exporter.
- Contexts are not included in the SFT messages; the generated dataset does
  not reproduce the runtime RAG context unless a future converter adds it.
- No fine-tuning, DPO, reward modeling, adapter management, or model serving
  code exists in the repository.

Recommended future placement:

- Keep Firestore collection and record normalization under
  `src/training_data/`.
- Keep one-shot dataset exports, format converters, and validation under
  `scripts/`.
- Put training configurations and executable training orchestration in a new
  top-level `training/` area, or a separately owned training repository. Keep
  it outside the runtime `src/llm/` and `src/rag_flow/` packages.
- Keep evaluation data and evaluation runners under `data/eval/` and a
  separately owned `training/eval/` or `scripts/` module.

## Configuration and Environment Audit

### Settings currently defined

`configs/settings.py` loads `.env` with `load_dotenv(override=True)` and
defines data paths, Firebase settings, OpenAI/Gemini credentials and model
names, embedding settings, retrieval thresholds, and chunking values.

The current `.env` selects Gemini, supplies a Gemini API credential, sets the
embedding/chunking/retrieval values, and enables Firebase. Its secret value is
intentionally omitted from this report. `.env` and
`firebase-service-account.json` are ignored by `.gitignore`.

### Problems found

1. `.env.example` does not exist. The only example configuration is embedded
   in documentation, so setup and deployment variables are not represented by
   a versioned template.
2. `LOCAL_LLM_URL` is not defined in `configs/settings.py`, despite being
   accessed by the `local` branch of `LLMFactory`. Selecting `LLM_PROVIDER=local`
   will fail during client creation with a missing settings attribute.
3. `LOCAL_LLM_MODEL` is neither defined nor read anywhere. The current model
   selection code falls back to `GEMINI_MODEL` for `local`.
4. Provider normalization/validation is absent. Whitespace, casing, and
   unsupported values are handled inconsistently or fail late.
5. Model selection and provider metadata are duplicated in the two application
   entry paths and in Streamlit logging. A future local provider can be
   configured successfully in the factory while still being called with, or
   logged as, the wrong model name.
6. `src/training_data/firebase_store.py` reads Firebase environment variables
   directly instead of using the `settings` object. This creates two
   configuration access patterns. Relative credential paths also depend on the
   process working directory.
7. `load_dotenv(override=True)` allows repository-local `.env` values to
   override deployment environment values. This should be an explicit
   deployment decision before production use.
8. The local `.env` contains a credential-like API key. The value is not
   tracked according to the current ignore rules, but it must remain secret;
   rotate it if it has been exposed beyond the trusted environment.
9. `docker-compose.yaml` forwards only `OPENAI_API_KEY`; it does not forward
   the selected provider, model settings, Gemini credential, Firebase settings,
   or local endpoint settings. Container behavior therefore does not match the
   documented `.env`-driven local behavior.
10. The Docker configuration has inconsistent port signals: the image exposes
    7860, the compose service maps 8000, and the Docker command starts Uvicorn
    on 8000. This is separate from local LLM integration but matters for
    serving a local endpoint or API deployment.

### Recommended local integration points

The future local-provider change should be isolated to these contracts:

1. Add `LOCAL_LLM_URL` and `LOCAL_LLM_MODEL` to `configs/settings.py` and a
   versioned `.env.example`. Decide and document whether a local API key is
   optional or configurable.
2. Update `src/llm/model_factory.py` to use the configured local base URL and
   preserve the existing `AsyncOpenAI` client contract.
3. Centralize model-name selection so `streamlit_app.py`,
   `src/api/endpoint.py`, and logging all use the same provider/model mapping.
4. Verify that the selected local server supports
   `chat.completions.create(..., stream=True)` and the response chunk shape
   consumed by both callers.
5. Add provider-selection tests using a mocked client/configuration. These
   tests should not require Ollama, a downloaded model, or a running server.

No RAG or prompt rewrite is required merely to connect an OpenAI-compatible
local inference server. RAG should remain the source of current legal
knowledge; model training should target response style/format and other
behavioral patterns rather than replace retrieval of changing legal content.

## Conflict Risk and Parallel Work

### High conflict risk

- `streamlit_app.py`: combines UI, routing, model invocation, logging,
  feedback, and review. Local-provider, feedback, and UI tasks all touch this
  file.
- `src/llm/model_factory.py`: central provider construction point.
- `configs/settings.py`: shared by ingestion, RAG, both app paths, logging, and
  future local configuration.
- `.env` / future `.env.example`: shared operational contract; secrets must not
  be merged casually.
- `src/training_data/firebase_store.py`: persistence behavior, feedback
  updates, review status, and exporter reads all depend on it. It also owns the
  effective Firestore schema contract.
- `src/llm/prompt_templates.py`: shared prompt behavior and likely target of
  future training/prompt-alignment work.
- `src/api/endpoint.py`: separate runtime path that duplicates provider/model
  selection and may be changed alongside Streamlit.
- `requirements.txt`: local inference, training, and runtime dependency work
  can create overlapping changes.

### Modules that can be developed independently

- A mocked provider-selection test suite, provided it does not change runtime
  code or provider behavior.
- Dataset validation and anonymizer tests under `tests/`, using fixture records
  instead of Firebase.
- New dataset format converters under `scripts/` with inputs/outputs passed as
  paths.
- Training configs and training runners under a new `training/` directory,
  with a documented JSONL input contract.
- Evaluation runners and additional fixtures under `data/eval/` plus an
  isolated evaluation module.
- Documentation updates, including `.env.example` documentation and provider
  compatibility notes.
- A new local provider adapter module, if the factory remains the only runtime
  integration point and shared model selection is coordinated separately.

The safest ownership split is one owner for the runtime/config contract
(`settings`, factory, and callers), one owner for Firebase/data contracts, and
one owner for independent dataset/training tooling.

## Existing Tests and Verification Snapshot

The repository currently contains one test module:
`tests/test_rag_optimization.py`.

It tests:

- expected embedding model and chunk size;
- prompt interpolation and a specific prompt phrase;
- unsupported document extension handling.

Current execution result from `pytest -q`: `1 passed, 2 failed`.

The failures are pre-existing expectation drift:

- the test expects `CHUNK_SIZE == 300`, while current settings/default and
  `.env` use `2500`;
- the test expects a prompt phrase that is not present in the current
  `SYSTEM_PROMPT`.

There are no tests for `LLMFactory`, provider selection, local settings,
Firebase persistence, feedback/review updates, anonymization, dataset export,
FastAPI streaming, or the Streamlit flow. No test changes were made for this
audit.

## File Inventory

| Area | Files | Responsibility |
| --- | --- | --- |
| Streamlit | `streamlit_app.py` | Chat UI, emergency/no-data/clarification/LLM branches, logging, rating, review UI |
| FastAPI/static | `main.py`, `src/api/endpoint.py`, `static/index.html`, `static/script.js`, `static/style.css` | Alternate SSE chatbot path and browser UI |
| RAG | `src/rag_flow/*.py` | Query analysis, retrieval, grading, reasoning gate |
| Ingestion | `ingestion_pipeline.py`, `src/ingestion/*.py` | Load raw documents, chunk, extract metadata, build FAISS artifacts |
| Prompt/LLM | `src/llm/prompt_templates.py`, `src/llm/model_factory.py` | Prompt template and OpenAI-compatible client factory |
| Configuration | `configs/settings.py`, `.env`, `.gitignore`, `docker-compose.yaml`, `Dockerfile` | Runtime values, secret exclusion, container environment |
| Firebase/data | `src/training_data/collector.py`, `firebase_store.py`, `anonymizer.py` | Record schema, Firestore operations, PII redaction |
| Dataset scripts | `scripts/export_sft_dataset.py`, `scripts/export_preference_dataset.py`, `scripts/validate_training_exports.py` | Export and validate JSONL datasets |
| Data artifacts | `data/processed/`, `data/training/exports/`, `data/eval/` | Retrieval index, chunks, training exports, evaluation questions |
| Tests | `tests/test_rag_optimization.py` | Current narrow unit coverage |

## Conclusion

The current architecture already has the right high-level seam for a future
local inference server: keep retrieval and prompt construction unchanged,
construct an OpenAI-compatible async client through `LLMFactory`, and select a
provider-specific model consistently in both runtime paths. The immediate
integration blockers are missing local configuration fields and duplicated
model selection. The separate training boundary is also present, but
preference-export status filtering, export validation coverage, and runtime
context representation should be resolved before using exports for DPO or
other preference training.

This task stops at architecture analysis. No local model was installed or
downloaded, no training was run, and application behavior was not modified.
