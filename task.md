# Task Plan: Thu thap Q&A de retrain va hoc tang cuong

Muc tieu: them pipeline thu thap cau hoi/cau tra loi tu Streamlit, luu len Firebase Firestore, co feedback/review, va export du lieu cho fine-tuning/preference learning.

Nguyen tac:
- Khong commit `firebase-service-account.json`.
- Firebase loi khong duoc lam chatbot dung.
- Khong train truc tiep tu raw logs.
- Chi export du lieu da review va da redact PII.

## Thu tu thuc hien

```text
T1 -> T2 -> T3 -> T4 -> T5
```

Co the chay song song:
- Sau `T2`, co the tach `T3` va `T4` cho 2 agent khac nhau neu kiem soat conflict tot.

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

## T2 - Logging Q&A len Firebase

Uu tien: P0

Phu thuoc: T1

Muc tieu: moi cau hoi/cau tra loi tren Streamlit duoc luu len Firestore.

File lien quan:
- `src/training_data/__init__.py`
- `src/training_data/firebase_store.py`
- `src/training_data/collector.py`
- `streamlit_app.py`

Prompt cho agent:

```text
Them pipeline logging Q&A len Firebase Firestore.

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
  - tao UUID neu thieu session_id/turn_id
  - contexts chi luu text, metadata, score
- Gan vao `streamlit_app.py`:
  - tao session_id co dinh trong st.session_state
  - log du cac nhanh: normal, clarification, no_data, emergency, error
  - khong doi flow tra loi hien tai
  - neu save thanh cong thi luu `last_logged_turn_id`
```

Done when:
- Hoi bot tren Streamlit, Firestore co document moi trong `chat_logs`.
- Chatbot van chay neu Firebase tat hoac loi.

---

## T3 - Feedback va review noi bo

Uu tien: P1

Phu thuoc: T2

Muc tieu: co nut danh gia cau tra loi va man hinh review/sua du lieu truoc khi train.

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
- Tranh gui feedback nhieu lan cho cung turn trong cung session.
- Them sidebar selectbox: Chatbot / Review data.
- Review data hien thi logs tu Firebase:
  - user_query
  - assistant_answer
  - response_type
  - feedback
  - contexts filename/score
- Cho admin nhap corrected_answer, error_type.
- Cho chon review_status: raw/approved/rejected/edited.
- Luu review_status len Firebase.
```

Done when:
- Feedback cap nhat duoc tren Firestore.
- Admin co the approved/rejected/edited mot log.

---

## T4 - Redact PII va export dataset train

Uu tien: P2

Phu thuoc: T3

Muc tieu: tao duoc dataset fine-tuning va preference learning tu du lieu da review.

File lien quan:
- `src/training_data/anonymizer.py`
- `scripts/export_sft_dataset.py`
- `scripts/export_preference_dataset.py`
- `data/training/exports/`

Prompt cho agent:

```text
Them anonymizer va script export dataset train.

Yeu cau anonymizer:
- Tao `src/training_data/anonymizer.py`
- Ham `redact_pii(text: str) -> str`
- Redact:
  - so dien thoai Viet Nam -> [PHONE]
  - CCCD/CMND 9-12 so -> [ID_NUMBER]
  - email -> [EMAIL]
  - ma ho so dang chu+so dai -> [CASE_ID]
- Ham `redact_record(record: dict) -> dict`

Yeu cau export SFT:
- Tao `scripts/export_sft_dataset.py`
- Doc logs tu Firebase
- Chi lay review_status approved hoac edited
- Neu edited va co corrected_answer thi dung corrected_answer
- Redact PII truoc khi export
- Output `data/training/exports/sft_dataset.jsonl`
- Format:
  {"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}

Yeu cau export preference:
- Tao `scripts/export_preference_dataset.py`
- Doc logs tu Firebase
- Chi tao sample khi co corrected_answer
- chosen = corrected_answer
- rejected = assistant_answer ban dau
- Redact PII
- Output `data/training/exports/preference_dataset.jsonl`
- Format:
  {"prompt":"...","chosen":"...","rejected":"...","reason":"..."}
```

Done when:
- Export duoc `sft_dataset.jsonl`.
- Export duoc `preference_dataset.jsonl`.
- Du lieu export khong con PII raw co ban.

---

## T5 - Demo docs va validate dataset

Uu tien: P2

Phu thuoc: T2 cho demo docs, T4 cho validate

Muc tieu: co tai lieu demo cho sep va script kiem tra dataset.

File lien quan:
- `docs/firebase_demo.md`
- `data/eval/eval_questions.jsonl`
- `scripts/validate_training_exports.py`

Prompt cho agent:

```text
Them tai lieu demo Firebase va cong cu validate dataset.

Yeu cau docs:
- Tao `docs/firebase_demo.md`
- Huong dan:
  - tao Firebase project
  - bat Firestore
  - tao service account
  - dat `firebase-service-account.json` local
  - cau hinh env
  - chay Streamlit
  - hoi bot va kiem tra Firestore collection `chat_logs`
  - khong commit private key

Yeu cau eval:
- Tao `data/eval/eval_questions.jsonl`
- 10 cau mau gom PCCC, dang ky xe, cu tru, thu tuc hanh chinh, no_data, emergency
- Moi dong co question, expected_points, source_hint, category

Yeu cau validate:
- Tao `scripts/validate_training_exports.py`
- Kiem tra SFT/preference JSONL hop le
- Kiem tra SFT co role user/assistant
- Kiem tra preference co prompt/chosen/rejected
- Regex canh bao neu con phone/email/CCCD raw
- In summary so mau hop le/loi/canh bao
```

Done when:
- Co doc demo cho sep.
- Co eval questions.
- Co script validate export.

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
- Sep khong can cai database hay tool phu.

## Milestone retrain/RL day du

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
- Co feedback.
- Co review/corrected answer.
- Co SFT dataset cho fine-tuning.
- Co preference dataset cho DPO/RLHF/RLAIF sau nay.
