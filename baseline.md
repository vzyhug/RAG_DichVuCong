# Báo cáo Tổng hợp Quá trình Thực hiện Dự án DVC-BCA-RAG

Dự án **DVC-BCA-RAG** là hệ thống Chatbot ứng dụng công nghệ Retrieval-Augmented Generation (RAG) nhằm hỗ trợ người dân tra cứu thông tin về thủ tục hành chính, pháp luật thuộc thẩm quyền của Công an phường/xã (ví dụ: cư trú, căn cước, PCCC, đăng ký xe, v.v.).

Dưới đây là tổng hợp toàn bộ các bước triển khai từ khâu chuẩn bị, xử lý dữ liệu, xây dựng Baseline, cho đến các bước tối ưu hóa.

---

## 1. Chuẩn bị (Preparation)
- **Công nghệ lõi:** Python, FastAPI (ban đầu) và Streamlit (giao diện frontend hiện tại).
- **Thư viện xử lý dữ liệu (Document Loaders):** `pypdf`, `pdfplumber`, `docx2txt`, `bs4`, `unidecode`.
- **Thư viện AI & Vector Search:** `sentence-transformers` (tạo text embeddings), `faiss-cpu` (lưu trữ và tìm kiếm vector), `openai` (kết nối LLM qua API OpenAI hoặc Gemini).
- **Cấu trúc thư mục:** Dữ liệu thô lưu tại `data/raw`, sau khi xử lý lưu index và chunk tại thư mục được cấu hình.

---

## 2. Xử lý dữ liệu (Data Ingestion Pipeline)
Quá trình đưa tài liệu (PDF, Word, Text) vào hệ thống được tự động hóa qua file `ingestion_pipeline.py` với các bước:
1. **Load Documents:** Đọc toàn bộ các văn bản pháp luật, tài liệu hướng dẫn thủ tục hành chính từ thư mục thô.
2. **Chunking (Chia nhỏ văn bản):** Phân rã các tài liệu lớn thành các đoạn văn bản (chunks) nhỏ hơn (ví dụ: 300-500 từ). Điều này giúp nhúng (embed) chính xác hơn và không vượt quá giới hạn token của LLM.
3. **Lưu trữ Metadata:** Mỗi chunk đều chứa metadata về nguồn tài liệu, giúp trích dẫn nguồn cho người dùng. File chunks được lưu dưới dạng `.jsonl`.
4. **Embedding & Indexing:** 
   - Dùng mô hình Embedding (như `intfloat/multilingual-e5-*` hoặc `keepitreal/vietnamese-sbert`) để biến các chunks văn bản thành các vector.
   - Xây dựng chỉ mục tìm kiếm bằng **FAISS** và lưu xuống ổ cứng để truy xuất cực nhanh.

---

## 3. Kiến trúc Baseline (Luồng RAG Cơ bản)
Luồng xử lý khi người dùng đặt câu hỏi (được triển khai trong backend và `streamlit_app.py`):

1. **Routing & Guardrails (Kiểm tra khẩn cấp):** 
   - Phân tích từ khóa để nhận diện các tình huống khẩn cấp (cháy nổ, cấp cứu, tội phạm).
   - Nếu là khẩn cấp, bot trả lời ngay lập tức số điện thoại trực ban (113, 114, 115) mà không cần gọi LLM, đảm bảo an toàn.
2. **Retrieval (Truy xuất ngữ cảnh):** 
   - Mã hóa câu hỏi của người dùng thành vector qua thư viện `ContextRetriever`.
   - Truy vấn trong FAISS để lấy ra Top-K chunks có độ tương đồng ngữ nghĩa cao nhất.
3. **Reasoning (Đánh giá thông tin):**
   - Đánh giá xem câu hỏi có đủ ý nghĩa để truy xuất không (`ReasoningChain`).
   - Nếu dữ liệu FAISS không trả về thông tin khớp, bot từ chối trả lời để tránh tình trạng ảo giác (Hallucination).
4. **Generation (Sinh văn bản):**
   - Nối các chunks lại làm ngữ cảnh (Context) và đưa vào Prompt Template cùng câu hỏi.
   - Gửi Prompt tới LLM (OpenAI/Gemini) với tham số `temperature=0.3` để đảm bảo tính khách quan và văn phong hành chính.
   - LLM sinh câu trả lời dưới dạng Stream (từng chữ một) và hiển thị trực tiếp lên giao diện Streamlit/FastAPI, kèm theo danh sách tài liệu trích dẫn rành mạch.

---

## 4. Tối ưu hóa hệ thống (RAG Optimization)
Dựa trên phân tích (từ file `rag_optimization_analysis.md`), các điểm yếu của Baseline đã được cải thiện:

### 4.1. Vấn đề LLM trả về tên file PDF thay vì câu trả lời
- **Nguyên nhân:** Khả năng trích xuất văn bản từ PDF scan kém, hoặc System Prompt chưa đủ nghiêm ngặt.
- **Giải pháp:** 
  - Bổ sung `pdfplumber` (hoặc dùng OCR) để đọc PDF tốt hơn, chống lỗi rỗng chunk.
  - Tối ưu lại System Prompt: *"You MUST directly answer the question using the provided context. DO NOT ask the user to read or download the source files."*
  - Xử lý lỗi UI (ẩn badge nguồn tài liệu nếu sinh văn bản bị lỗi/rỗng).

### 4.2. Vấn đề tốc độ phản hồi chậm (High Latency)
- **Nguyên nhân:** Mô hình embedding quá nặng, chunk size quá lớn, hoặc API request bị chặn đứng đến khi load xong (block API).
- **Giải pháp:**
  - **Streaming Output:** Áp dụng luồng Stream cho LLM, giúp frontend hiển thị chữ ngay lập tức khi LLM đang suy nghĩ.
  - **Mô hình Embedding nhẹ hơn:** Chuyển từ các mô hình lớn (2.2GB) sang `vietnamese-sbert` hoặc e5-small để tăng tốc CPU inference.
  - **Tối ưu ngữ cảnh:** Thu gọn `CHUNK_SIZE` để giảm tải Token cho LLM, qua đó giảm thời gian sinh token của LLM.

---
**Kết luận:** Dự án đã phát triển từ một kiến trúc Backend cơ bản với RAG pipeline thô sang một giải pháp hoàn chỉnh, có giao diện dễ sử dụng bằng Streamlit, Indexing tự động, cùng với khả năng sinh luồng và trích xuất nguồn văn bản đáng tin cậy.
