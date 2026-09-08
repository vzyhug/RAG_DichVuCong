# 👮‍♂️ Chatbot Trợ Lý Ảo - Công an xã An Viễn

Chào mừng bạn đến với dự án Chatbot Trợ Lý Ảo của Công an xã An Viễn! Đây là một giải pháp công nghệ nhằm kéo gần khoảng cách giữa lực lượng công an và người dân, giúp mọi người tra cứu thông tin pháp luật và thủ tục hành chính một cách dễ dàng và thân thiện nhất.

## 🌟 Chức năng nổi bật
- **📝 Hướng dẫn thủ tục siêu chi tiết:** Chỉ cần người dân đặt câu hỏi, Chatbot sẽ trả lời ngay từng bước thực hiện, hồ sơ cần chuẩn bị, lệ phí và thời gian giải quyết với bố cục cực kỳ rõ ràng, dễ đọc.
- **🚨 Trợ lý khẩn cấp:** Hướng dẫn người dân quy trình báo án nhanh chóng, cung cấp ngay số điện thoại đường dây nóng trực ban khi có sự cố (tai nạn, trộm cắp, cháy nổ).
- **📞 Tra cứu nhanh danh bạ:** Cần tìm ai, số điện thoại nào trong ban chỉ huy hoặc cán bộ xã? Hỏi Chatbot là có ngay!
- **🧠 Nói không với "chém gió":** Chatbot chỉ trả lời dựa trên kho dữ liệu pháp luật và tài liệu nội bộ đã được cung cấp. Nếu không biết, nó sẽ từ chối khéo léo chứ tuyệt đối không bịa thông tin.

## 🛠 Hệ thống hoạt động ra sao?
Dự án được xây dựng tối giản để ai cũng có thể hiểu và cài đặt dễ dàng:
- **Giao diện trò chuyện:** Thiết kế bằng Streamlit, cực kỳ thân thiện và mượt mà.
- **Bộ não thông minh:** Tận dụng sức mạnh của Google Gemini AI để đọc hiểu câu hỏi và tự động soạn thảo câu trả lời trôi chảy.
- **Kho lưu trữ trí nhớ:** Bất kỳ tài liệu luật hay hướng dẫn nào (PDF, Word) cũng được máy tính tự động đọc hiểu, băm nhỏ và lưu vào bộ nhớ FAISS. Nhờ đó, AI có thể bới tìm chính xác thông tin chỉ trong chớp mắt.

## 🚀 Hướng dẫn trải nghiệm trên máy tính (Local)

1. **Cài đặt thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Cấp chìa khóa cho AI hoạt động:**
   Tạo file `.env` ở thư mục gốc và dán thông số sau vào:
   ```env
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL=gemini-3.5-flash-lite
   EMBEDDING_MODEL=intfloat/multilingual-e5-small
   TOP_K=10
   CHUNK_SIZE=2500
   CHUNK_OVERLAP=400
   SIMILARITY_THRESHOLD=0.70
   ```

3. **Nạp kiến thức mới cho Chatbot:**
   Bạn có thủ tục mới? Chỉ việc thả file (PDF, Word, Text) vào thư mục `data/raw/` rồi chạy lệnh:
   ```bash
   python ingestion_pipeline.py
   ```

4. **Khởi động Chatbot:**
   ```bash
   streamlit run streamlit_app.py
   ```

## ☁️ Đưa Chatbot lên mạng (Triển khai lên Hugging Face)
Ứng dụng này cực kỳ nhẹ và có thể chạy tẹt ga 24/7 hoàn toàn miễn phí trên Hugging Face:
1. Tạo một **Streamlit Space** mới trên Hugging Face.
2. Cấu hình máy chủ: Chọn loại **CPU basic** (Miễn phí).
3. Vào mục **Settings > Variables and secrets**, thêm biến `GEMINI_API_KEY`.
4. Upload toàn bộ thư mục code (nhớ upload kèm thư mục `data/processed` để Chatbot có sẵn kiến thức) lên Space là xong! Ứng dụng sẽ tự động khởi chạy để phục vụ người dân.