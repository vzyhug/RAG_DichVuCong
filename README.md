# DVC-BCA-RAG - Chatbot Công an phường/xã

Hệ thống RAG (Retrieval-Augmented Generation) dành cho chatbot hỗ trợ người dân tại Công an phường/xã, dựa trên kho dữ liệu pháp lý và hướng dẫn thủ tục hành chính.

## Tính năng
- Trả lời câu hỏi về thủ tục hành chính (cư trú, căn cước, đăng ký xe, PCCC, ...)
- Nhận diện tình huống khẩn cấp và hướng dẫn gọi số 113/114/115
- Hỏi làm rõ khi thiếu thông tin
- Tuân thủ các quy tắc an toàn (không yêu cầu OTP, không kết luận hồ sơ)

## Cài đặt và chạy

1. **Clone repository**
2. **Cài đặt môi trường ảo và dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt