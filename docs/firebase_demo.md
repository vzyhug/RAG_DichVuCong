# Firebase Demo Guide

This guide is for a local demonstration of the An Vien commune police assistant. The app stores each question, answer, and retrieved RAG chunks as a document in the Firestore `chat_logs` collection when Firebase logging is enabled.

The collected Q&A logs are used to review and improve chunking, metadata, retrieval thresholds, source coverage, and re-indexing decisions. They are not intended for direct Gemini API fine-tuning. Preference/RL exports are still supported when an admin supplies a `corrected_answer`.

## 1. Create a Firebase project

1. Open the [Firebase console](https://console.firebase.google.com/).
2. Select **Add project**.
3. Enter a project name, continue through the setup prompts, and create the project.
4. Keep the project open; it will be used when creating the service-account key.

## 2. Enable Firestore

1. In the Firebase console, open **Build > Firestore Database**.
2. Select **Create database**.
3. Choose a location close to the demo environment.
4. Start in production mode unless this is an isolated local demo. The server-side service account is used by the app, so do not make the database publicly readable or writable.

## 3. Create a service account key

1. Open **Project settings > Service accounts**.
2. Select **Generate new private key** and confirm.
3. Download the JSON file.
4. Rename it to `firebase-service-account.json` and place it in the repository root for a local demo.

The downloaded file is a private credential. Do not commit it, upload it to a public repository, paste it into a ticket, or include it in a screenshot. This repository ignores `firebase-service-account.json`; verify that it is not shown by `git status` before sharing changes. Delete or rotate the key after a demo if it was exposed.

## 4. Configure environment variables

From the repository root, create or update `.env`:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite

ENABLE_FIREBASE_LOGGING=true
FIREBASE_CREDENTIALS_PATH=firebase-service-account.json
FIREBASE_COLLECTION=chat_logs
FIREBASE_DATA_UPDATE_COLLECTION=data_update_requests

GITHUB_TOKEN=your_github_token
GITHUB_REPOSITORY=owner/repository
GITHUB_DATA_UPDATE_LABELS=data-update,rag,needs-review
GITHUB_PROJECT_ID=your_project_v2_node_id
```

Install the dependencies before running the demo:

```bash
pip install -r requirements.txt
```

For a key stored outside the repository, set `FIREBASE_CREDENTIALS_PATH` to its local path instead. Keep API keys and the service-account JSON out of source control.

GitHub configuration is optional for chat logging, but required if the admin uses **Gửi đội kỹ thuật xử lý dữ liệu** from the review screen. `GITHUB_TOKEN` needs permission to create issues in `GITHUB_REPOSITORY`. If `GITHUB_PROJECT_ID` is configured, the app also tries to add the created issue to that GitHub Projects v2 board. Without `GITHUB_PROJECT_ID`, keep using Project automation rules based on the labels in `GITHUB_DATA_UPDATE_LABELS`.

## 5. Run Streamlit

Run this command from the repository root:

```bash
streamlit run streamlit_app.py
```

Streamlit prints a local URL, normally `http://localhost:8501`. Open it in a browser.

Use the sidebar in Streamlit to switch between the chatbot and **Dữ liệu review**. The review screen lets an admin inspect collected Q&A logs and send data-fix requests to the technical team.

## 6. Run the demo and verify Firestore logging

1. Ask the assistant a question such as: `Thủ tục đăng ký xe máy lần đầu gồm những gì?`
2. Wait for the answer to finish.
3. In the Firebase console, open **Build > Firestore Database > Data**.
4. Open the `chat_logs` collection. A new document should contain fields such as `user_query`, `assistant_answer`, `response_type`, `contexts`, `session_id`, `turn_id`, `created_at`, and `review_status`.
5. Use `created_at` or `user_query` to identify the document from the demo turn.

The `contexts` field is the main evidence for data review. It shows which source snippets were retrieved, their metadata, and their scores. During review, the admin should use non-technical choices such as missing data, wrong data, outdated data, incomplete answer, or wrong source. If **Gửi đội kỹ thuật xử lý dữ liệu** is selected, the app creates a document in `data_update_requests`, creates a GitHub Issue for developer follow-up, and adds the issue to GitHub Projects when `GITHUB_PROJECT_ID` is available.

Logging is best-effort: if Firebase is disabled, unavailable, or misconfigured, the chatbot can still answer. For a demo, check the Streamlit terminal for warnings if no document appears, then confirm the credential path, `ENABLE_FIREBASE_LOGGING=true`, the selected Firebase project, and Firestore access.

## Key handling checklist

- Keep `firebase-service-account.json` only on the local machine or in a secret manager.
- Never commit the private key or add it to a deployment artifact.
- Use environment/secrets settings for hosted deployments.
- Rotate the service-account key immediately if it is exposed.
