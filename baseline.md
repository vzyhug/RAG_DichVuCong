# Tổng hợp quá trình thực hiện: Từ Tiền xử lý dữ liệu đến Baseline Model (LegalQA)

Dưới đây là tóm tắt toàn bộ các bước đã thực hiện trong bài toán LegalQA, được chia thành hai giai đoạn chính: **Tiền xử lý dữ liệu (Preprocessing)** và **Xây dựng mô hình cơ sở (Baseline RAG Pipeline)**.

---

## 1. Giai đoạn Tiền xử lý dữ liệu (Preprocessing)
Quá trình này được thực hiện trong file [`preprocessing_pipeline.ipynb`](file:///E:/nam4_hk1/dsc/preprocessing_pipeline.ipynb) nhằm làm sạch và chuẩn bị dữ liệu văn bản pháp luật từ thư mục `selected-contexts`.

### 1.1. Làm sạch văn bản (Text Cleaning)
- **Loại bỏ Boilerplate:** Xóa bỏ các thành phần rập khuôn không mang nhiều ý nghĩa tra cứu như Quốc hiệu, tiêu ngữ ("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Độc lập - Tự do - Hạnh phúc"), số hiệu văn bản, và ngày tháng ban hành.
- **Sửa lỗi ngắt dòng:** Nối lại các dòng bị ngắt sai ngữ pháp (những dòng không kết thúc bằng dấu chấm, phẩy, hai chấm, hỏi chấm, chấm cảm).
- **Loại bỏ khoảng trắng thừa:** Chuẩn hóa các dấu xuống dòng liên tiếp (đưa về tối đa 2 lần xuống dòng) và xóa các khoảng trắng dư thừa trong câu.

### 1.2. Chuẩn hóa định dạng (Heuristic Formatting)
- Các dòng có cấu trúc giống công thức hoặc bảng biểu được điều chỉnh lại khoảng trắng xung quanh dấu `=` để văn bản được đồng nhất hơn.

### 1.3. Phân rã văn bản & Bổ sung ngữ cảnh (Semantic Chunking & Metadata Injection)
- **Chunking:** Tách các văn bản pháp luật dài thành từng đoạn nhỏ (chunk), sử dụng dấu hiệu nhận biết là các từ khóa "Điều X." hoặc "Điều X:".
- **Metadata Injection:** Thêm tên/tiêu đề của tài liệu (doc_title) vào đầu mỗi chunk (`[doc_title]\n{nội dung}`) để mô hình RAG sau này luôn nắm được đoạn văn bản trích xuất đang thuộc về văn bản luật nào.

### 1.4. Xuất dữ liệu
- Dữ liệu sau khi xử lý được lưu dưới dạng file `processed_contexts.jsonl`. Mỗi dòng chứa thông tin của một chunk (`doc_id`, `title`, `chunk_text`), tối ưu cho quá trình nạp dữ liệu ở bước sau.

---

## 2. Giai đoạn Mô hình Cơ sở (Baseline RAG Pipeline)
Giai đoạn này được triển khai trong file [`baseline_legalqa.ipynb`](file:///E:/nam4_hk1/dsc/baseline_legalqa.ipynb), ứng dụng kiến trúc **RAG (Retrieval-Augmented Generation)** để sinh câu trả lời.

### 2.1. Xây dựng bộ truy xuất (Retriever) với BM25
- **Tải và Tokenize dữ liệu:** Đọc file `processed_contexts.jsonl`. Văn bản tiếng Việt trong các chunk được tách từ (word segmentation) bằng thư viện `pyvi` (`ViTokenizer`).
- **Khởi tạo BM25:** Sử dụng thuật toán `BM25Okapi` để lập chỉ mục các chunk đã tokenize.
- **Hàm truy xuất:** Khi có câu hỏi, truy vấn cũng được tokenize và BM25 sẽ trả về `top_k=3` đoạn ngữ cảnh phù hợp nhất.

### 2.2. Khởi tạo mô hình ngôn ngữ (Generator)
- **Mô hình sử dụng:** LLM `thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2` (mô hình đã được tinh chỉnh cho tiếng Việt và lĩnh vực pháp lý).
- **Tối ưu phần cứng:** Mô hình được load bằng kiểu dữ liệu `Float16` (`torch.float16`) và sử dụng `device_map="auto"` để tự động phân bổ lên GPU (T4 trên Colab), giúp tiết kiệm VRAM.

### 2.3. Quy trình Hỏi đáp (RAG Pipeline)
- **Gộp ngữ cảnh:** Nối top 3 chunks tìm được từ BM25 thành một chuỗi văn bản hoàn chỉnh.
- **Xây dựng Prompt:** Tạo lời nhắc (prompt) đóng vai chuyên gia pháp lý, cung cấp ngữ cảnh đã gộp và yêu cầu trả lời ngắn gọn, chính xác câu hỏi. Nếu không có thông tin, mô hình được chỉ thị trả lời "Không tìm thấy thông tin".
- **Sinh văn bản:** Chạy mô hình với các tham số khống chế như `max_new_tokens=256`, `temperature=0.3`, và `repetition_penalty=1.1` để câu trả lời mang tính ổn định, ít ngẫu nhiên và không bị lặp từ.

### 2.4. Đánh giá và Xuất kết quả
- Pipeline được áp dụng lên tập dữ liệu test `public-official.json`.
- Duyệt qua từng câu hỏi, đưa vào mô hình để sinh câu trả lời tương ứng.
- Kết quả cuối cùng được xuất ra file `submission.json` theo format chuẩn của cuộc thi.
