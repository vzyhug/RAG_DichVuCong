FROM python:3.10-slim

# Cài đặt user không phải root (Yêu cầu bắt buộc của Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy thư mục hiện tại vào container với quyền của user
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

# Hugging Face Spaces mặc định sử dụng cổng 7860
EXPOSE 7860

CMD ["uvicorn", "src.api.endpoint:app", "--host", "0.0.0.0", "--port", "7860"]