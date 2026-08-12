# RAG System Optimization: PDF Outputs & Inference Speed

This document summarizes the causes and proposed solutions for two common issues observed in the current RAG (Retrieval-Augmented Generation) pipeline: the bot referencing `.pdf` files instead of providing text answers, and slow inference speeds.

---

## Issue 1: The Bot outputs a `.pdf` file instead of answering in text

### Cause 1: PDF Text Extraction Failure (Scanned Documents)
- **Analysis:** In `src/ingestion/document_loaders.py`, the system uses `pypdf` to extract text. If the uploaded PDFs are scanned documents or images, `pypdf` will fail to extract textual data, resulting in empty or corrupted context chunks. With no real text to process, the LLM cannot answer the query and might just regurgitate the document's filename.
- **Solution:** Implement OCR (Optical Character Recognition). Replace or supplement `pypdf` with robust libraries like `pdfplumber` or `pytesseract` (along with `pdf2image`) to accurately extract text from scanned pages.

### Cause 2: Weak System Prompt Instructions
- **Analysis:** The current `SYSTEM_PROMPT` in `src/llm/prompt_templates.py` instructs the bot to answer based on the context but does not explicitly forbid it from simply citing the source files. When faced with fragmented context, the LLM might lazily output *"Please refer to abc.pdf"*.
- **Solution:** Strengthen the System Prompt. Explicitly instruct the LLM to synthesize the information and directly answer the question. For example: *"You MUST directly answer the question using the provided context. DO NOT ask the user to read or download the source files."*

### Cause 3: UI Rendering Artifacts (Source Badges)
- **Analysis:** The frontend (`static/script.js`) automatically appends source document badges (e.g., `[📄 01.pdf]`) at the end of the bot's message based on retrieved metadata. If the LLM's actual text answer is empty (due to safety filters or API errors), the UI will only display the `.pdf` badge, giving the illusion that the bot only replied with a file.
- **Solution:** Implement strict backend logging for the LLM's raw output (e.g., `response.choices[0].message.content`) to monitor empty responses. Additionally, add fallback logic in the UI to display a default apology message if the generated answer is entirely empty.

---

## Issue 2: Slow Inference Speed (High Latency)

### Cause 1: Blocking API Calls (Lack of Streaming)
- **Analysis:** The `endpoint.py` server waits for the LLM to generate the entire response before sending the JSON payload back to the frontend. This causes a very high Time-To-First-Token (TTFT), making the bot feel unresponsive while the user stares at a typing indicator.
- **Solution:** Implement **Streaming Responses (Server-Sent Events - SSE)**. By passing `stream=True` to the OpenAI/Gemini client and wrapping the FastAPI endpoint in a `StreamingResponse`, the bot can stream the answer word-by-word instantly to the UI.

### Cause 2: Heavy Embedding Model
- **Analysis:** The retriever uses `intfloat/multilingual-e5-large`, which is computationally heavy (approx. 2.2GB). Running this model via `SentenceTransformer` for every user query adds significant latency, especially on CPU.
- **Solution:** Downgrade to a lighter, optimized embedding model such as `keepitreal/vietnamese-sbert` or `intfloat/multilingual-e5-small`. These provide comparable semantic search quality for Vietnamese with a fraction of the computational overhead.

### Cause 3: Large Context Window Overhead
- **Analysis:** Passing multiple large chunks (`CHUNK_SIZE=512`) to the LLM increases the input token count. The processing time of the LLM scales linearly (and sometimes quadratically) with the length of the prompt.
- **Solution:** Optimize the prompt size. Reduce `CHUNK_SIZE` to `300-400` words or dynamically adjust `top_k` based on similarity scores to ensure only highly relevant, concise context is sent to the LLM.
