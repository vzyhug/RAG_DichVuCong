# Task Plan: Thu thap Q&A de cai thien chunking/RAG va hoc tang cuong

Muc tieu: thu thap cau hoi/cau tra loi tu Streamlit, luu len Firebase Firestore, review loi truy xuat, roi dung log da duyet de cai thien chunking/re-index cua RAG. Khong fine-tune model API Gemini. Phan preference/RL van giu de tao cap chosen/rejected khi co cau tra loi da sua.

Nguyen tac:
- Khong commit `firebase-service-account.json`.
- Firebase loi khong duoc lam chatbot dung.
- Raw logs khong dung truc tiep de train hay re-index; phai review va redact PII.
- Q&A logs dung chinh de phat hien: sai chunk, thieu chunk, chunk qua ngan/dai, metadata kem, tai lieu nguon thieu/cu.
- Preference/RL chi dung voi record co `corrected_answer`.

## Thu tu thuc hien

```text
T1 -> T2 -> T3 -> T4 -> T5
```

Khong nen chay song song:
- Nhieu agent cung sua `streamlit_app.py`.
- Nhieu agent cung sua `src/training_data/firebase_store.py`.

---

## T1 - Firebase base config

Uu tien: P0

Muc tieu: project san sang ket noi Firebase, khong lo private key.

File lien quan:
- `.gitignore`
- `.env`
- `requirements.txt`
- `configs/settings.py` neu can

Prompt cho agent:

```text
Kiem tra va bo sung Firebase base config cho project.

Yeu cau:
- `requirements.txt` co `firebase-admin`.
- `.gitignore` ignore dung `firebase-service-account.json`.
- Khong track/commit Firebase private key.
- Them settings/env neu can:
  - ENABLE_FIREBASE_LOGGING, default false
  - FIREBASE_CREDENTIALS_PATH
  - FIREBASE_COLLECTION, default `chat_logs`
- Neu doc `.env`, khong in gia tri secret.
- Khong sua logic chatbot.
```

Done when:
- Dependency va ignore key dung.
- Khong co private key trong git status/staged.

---

## T2 - Logging Q&A va retrieved chunks len Firebase

Uu tien: P0

Phu thuoc: T1

Muc tieu: moi cau hoi/cau tra loi tren Streamlit duoc luu len Firestore kem chunks da retrieve.

File lien quan:
- `src/training_data/__init__.py`
- `src/training_data/firebase_store.py`
- `src/training_data/collector.py`
- `streamlit_app.py`

Prompt cho agent:

```text
Them pipeline logging Q&A len Firebase Firestore de phuc vu review chunking/RAG.

Yeu cau:
- Tao package `src/training_data/`.
- Tao `firebase_store.py`:
  - init firebase-admin an toan, tranh init nhieu lan
  - doc ENABLE_FIREBASE_LOGGING, FIREBASE_CREDENTIALS_PATH, FIREBASE_COLLECTION
  - ham `save_chat_log(record: dict) -> str | None`
  - disabled/thieu credentials/Firebase loi thi log warning va return None
- Tao `collector.py`:
  - ham `build_chat_record(...) -> dict`
  - ham `collect_chat_log(...) -> str | None`
  - record gom: session_id, turn_id, created_at, user_query, assistant_answer, response_type, contexts, entities, model, provider, review_status="raw", feedback=null, corrected_answer=null
  - contexts phai luu text, metadata, score de review chunk/re-index
- Gan vao `streamlit_app.py`:
  - tao session_id co dinh trong st.session_state
  - log du cac nhanh: normal, clarification, no_data, emergency, error
  - khong doi flow tra loi hien tai
```

Done when:
- Hoi bot tren Streamlit, Firestore co document moi trong `chat_logs`.
- Log co `contexts` de biet chunk nao da duoc lay.

---

## T3 - Feedback va review loi chunking/RAG

Uu tien: P1

Phu thuoc: T2

Muc tieu: co nut danh gia cau tra loi va man hinh review de gan nhan loi cho chunking/retrieval.

File lien quan:
- `src/training_data/firebase_store.py`
- `streamlit_app.py`

Prompt cho agent:

```text
Them feedback nguoi dung va review noi bo cho du lieu Q&A.

Yeu cau trong `firebase_store.py`:
- Them `update_chat_feedback(turn_id: str, feedback: dict) -> bool`
- Them `list_chat_logs(limit=100, review_status=None, rating=None) -> list[dict]`
- Them `update_review_status(turn_id, review_status, corrected_answer=None, error_type=None) -> bool`
- Firebase loi/disabled thi return False hoac list rong, khong crash app

Yeu cau trong `streamlit_app.py`:
- Sau moi cau tra loi co turn_id, hien thi feedback:
  - Huu ich -> rating up
  - Khong huu ich -> rating down
  - Sai thong tin -> rating wrong
- Them sidebar selectbox: Chatbot / Review data.
- Review data hien thi:
  - user_query
  - assistant_answer
  - response_type
  - feedback
  - contexts filename/score/text
- Cho admin nhap corrected_answer neu cau tra loi sai.
- Cho admin nhap error_type, uu tien cac nhan:
  - wrong_context
  - no_context
  - missing_context
  - incomplete
  - outdated_info
  - wrong_format
  - hallucination
- Cho chon review_status: raw/approved/rejected/edited.
```

Done when:
- Admin co the biet cau tra loi sai do chunk/retrieval hay do generation.
- Admin co the approved/rejected/edited mot log.

---

## T4 - Export chunk-review dataset va preference dataset

Uu tien: P2

Phu thuoc: T3

Muc tieu: tao duoc artifact de review chunking/re-index va artifact preference cho RL/DPO sau nay.

File lien quan:
- `src/training_data/anonymizer.py`
- `scripts/export_chunk_review_dataset.py`
- `scripts/export_preference_dataset.py`
- `data/training/exports/`

Prompt cho agent:

```text
Them anonymizer va script export du lieu review chunking.

Yeu cau anonymizer:
- Tao `src/training_data/anonymizer.py`
- Ham `redact_pii(text: str) -> str`
- Redact phone, CCCD/CMND, email, ma ho so.
- Ham `redact_record(record: dict) -> dict`

Yeu cau export chunk review:
- Tao `scripts/export_chunk_review_dataset.py`
- Doc logs tu Firebase
- Chi lay review_status approved/edited/rejected
- Redact PII
- Output `data/training/exports/chunk_review_dataset.jsonl`
- Moi row gom:
  - user_query
  - assistant_answer
  - corrected_answer
  - response_type
  - review_status
  - error_type
  - feedback
  - retrieved_contexts: text, metadata, score
  - chunking_action_hint

Yeu cau export preference:
- Tao `scripts/export_preference_dataset.py`
- Chi tao sample khi co corrected_answer
- chosen = corrected_answer
- rejected = assistant_answer ban dau
- Redact PII
- Output `data/training/exports/preference_dataset.jsonl`
```

Done when:
- Export duoc `chunk_review_dataset.jsonl` de phan tich chunk/re-index.
- Export duoc `preference_dataset.jsonl` khi co corrected_answer.

---

## T5 - Demo docs va validate exports

Uu tien: P2

Phu thuoc: T2 cho demo docs, T4 cho validate

Muc tieu: co tai lieu demo Firebase va script kiem tra export.

File lien quan:
- `docs/firebase_demo.md`
- `scripts/validate_training_exports.py`

Prompt cho agent:

```text
Them tai lieu demo Firebase va cong cu validate dataset.

Yeu cau docs:
- Tao `docs/firebase_demo.md`
- Huong dan tao Firebase project, bat Firestore, tao service account, dat key local, cau hinh env, chay Streamlit, xem `chat_logs`.
- Nhac ro Q&A logs dung de review chunking/re-index, khong fine-tune Gemini API.

Yeu cau validate:
- Tao `scripts/validate_training_exports.py`
- Kiem tra `chunk_review_dataset.jsonl` hop le.
- Kiem tra `preference_dataset.jsonl` hop le neu ton tai.
- Regex canh bao neu con phone/email/CCCD raw.
- In summary so mau hop le/loi/canh bao.
```

Done when:
- Co doc demo cho sep.
- Co script validate export chunk-review/preference.

---

## Milestone demo nhanh

Chi can:

```text
T1
T2
T5 docs phan Firebase demo
```

Ket qua:
- Streamlit chay.
- Hoi bot xong du lieu xuat hien tren Firebase Console.
- Sep thay duoc Q&A + retrieved chunks duoc luu de phan tich chunking.

## Milestone chunking/RL day du

Can:

```text
T1
T2
T3
T4
T5
```

Ket qua:
- Co Q&A logs.
- Co retrieved chunks trong tung cau hoi.
- Co error_type de biet can chunk lai, them tai lieu, sua metadata hay re-index.
- Co `chunk_review_dataset.jsonl` cho vong cai thien RAG.
- Co `preference_dataset.jsonl` cho DPO/RLHF/RLAIF neu co corrected_answer.
