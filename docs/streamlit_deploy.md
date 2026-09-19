# Streamlit Deployment

This app can be deployed as a Streamlit app with:

```text
streamlit_app.py
```

Do not upload `.env` or `firebase-service-account.json` to the repository. Put the values below in the deployment platform secrets instead.

## Required Secrets

```toml
GEMINI_API_KEY = "your_gemini_api_key"
GEMINI_MODEL = "gemini-3.5-flash-lite"

ENABLE_FIREBASE_LOGGING = "true"
FIREBASE_COLLECTION = "chat_logs"
FIREBASE_DATA_UPDATE_COLLECTION = "data_update_requests"
FIREBASE_SERVICE_ACCOUNT_JSON = '{"type":"service_account","project_id":"..."}'

GITHUB_TOKEN = "your_github_token"
GITHUB_REPOSITORY = "vzyhug/RAG_DichVuCong"
GITHUB_DATA_UPDATE_LABELS = "data-update,rag,needs-review"
GITHUB_PROJECT_ID = "your_project_v2_node_id"
```

`FIREBASE_SERVICE_ACCOUNT_JSON` must contain the full Firebase service-account JSON as one line. Generate it locally with:

```powershell
python -c "import json; print(json.dumps(json.load(open('firebase-service-account.json', encoding='utf-8'))))"
```

Copy the output into the secret value.

## Streamlit Community Cloud

1. Push the repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from the repository.
4. Set the main file path to:

```text
streamlit_app.py
```

5. Paste the secrets above into the app secrets.
6. Deploy.

## Verify After Deploy

1. Open the deployed app URL.
2. Ask the chatbot one question.
3. Confirm a new document appears in Firebase `chat_logs`.
4. Open `Dữ liệu review` in the sidebar.
5. Send one review to the technical team.
6. Confirm:
   - Firebase has a new `data_update_requests` document.
   - GitHub has a new Issue.
   - GitHub Project receives the Issue if the token has Project write access.

