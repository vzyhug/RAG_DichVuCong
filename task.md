Kiến Trúc Khi Đổi Sang Local LLM
  Project hiện đã có LLMFactory, nên về lý thuyết chỉ cần thêm provider local:

  Streamlit
  -> RAG retriever hiện tại
  -> build_prompt
  -> Local LLM server OpenAI-compatible
  -> trả lời

  Tức là không cần viết lại toàn bộ app. Chỉ cần chạy local model qua server có API tương thích OpenAI.

  Ví dụ:

  Ollama
  vLLM
  LM Studio
  Text Generation Inference
  llama.cpp server

  Project đã dùng kiểu OpenAI client:

  llm_client.chat.completions.create(...)

  Nên nếu server local hỗ trợ OpenAI-compatible API, chỉ cần config:

  LLM_PROVIDER=local
  LOCAL_LLM_URL=http://localhost:11434/v1
  LOCAL_LLM_MODEL=qwen2.5:7b-instruct

  Nhưng hiện src/llm/model_factory.py:10 đang gọi settings.LOCAL_LLM_URL, trong khi configs/settings.py chưa có biến này. Cần bổ sung.

  Model Nên Dùng
  Với tiếng Việt + pháp luật/thủ tục, ưu tiên:

  Qwen2.5/Qwen3 Instruct
  Llama 3.1/3.2 Instruct
  Mistral/Nemo Instruct

  Thực tế tiếng Việt, tôi sẽ ưu tiên:

  Qwen2.5-7B-Instruct
  Qwen2.5-14B-Instruct nếu máy đủ mạnh

  Nếu máy yếu:

  Qwen2.5-3B-Instruct

  Cấu Hình Theo Máy
  Nếu không có GPU mạnh:

  Ollama + Qwen 7B quantized

  Dễ demo, dễ chạy, nhưng fine-tune thật không chạy trên Ollama. Ollama chủ yếu để inference.

  Nếu có GPU:

  Fine-tune bằng LoRA/QLoRA
  Inference bằng vLLM hoặc llama.cpp/Ollama

  VRAM tham khảo:

  7B QLoRA: khoảng 12-16GB VRAM
  14B QLoRA: khoảng 24GB+ VRAM

  Luồng Chuyển Đổi Đề Xuất
  Giai đoạn 1: Chạy local model thay Gemini

  1. Cài Ollama hoặc LM Studio
  2. Pull model Qwen/Llama
  3. Bổ sung LOCAL_LLM_URL, LOCAL_LLM_MODEL trong settings
  4. Sửa LLMFactory để dùng local model name
  5. Test Streamlit

  Ví dụ với Ollama:

  ollama pull qwen2.5:7b-instruct
  ollama serve

  .env:

  LLM_PROVIDER=local
  LOCAL_LLM_URL=http://localhost:11434/v1
  LOCAL_LLM_MODEL=qwen2.5:7b-instruct

  Giai đoạn 2: Dùng dữ liệu Firebase để fine-tune

  Firebase logs
  -> Review data
  -> Export sft_dataset.jsonl
  -> Convert sang format Alpaca/ShareGPT nếu cần
  -> Fine-tune Qwen/Llama bằng LoRA/QLoRA

  Giai đoạn 3: Deploy model đã fine-tune

  adapter LoRA + base model
  -> merge hoặc load adapter
  -> serve bằng vLLM/llama.cpp/Ollama Modelfile
  -> app gọi LOCAL_LLM_URL

  Format Dataset
  Hiện project export dạng OpenAI messages:

  {
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }

  Format này dùng được hoặc convert được cho hầu hết tool fine-tune như:

  LLaMA-Factory
  Axolotl
  Unsloth
  TRL

  Nếu dùng LLaMA-Factory, có thể cần dataset dạng ShareGPT:

  {
    "conversations": [
      {"from": "system", "value": "..."},
      {"from": "human", "value": "..."},
      {"from": "gpt", "value": "..."}
    ]
  }

  Có thể viết script convert từ sft_dataset.jsonl sang ShareGPT.

  Fine-Tune Thật Nên Làm Bằng Gì
  Dễ nhất hiện nay:

  Unsloth
  LLaMA-Factory

  Khuyến nghị cho bạn:

  LLaMA-Factory + Qwen2.5-7B-Instruct + LoRA

  Vì:

  dễ cấu hình
  hỗ trợ nhiều model
  hỗ trợ SFT/DPO
  dùng được dataset ShareGPT/Alpaca
  phù hợp pipeline SFT + preference learning

  Học Tăng Cường Với Local LLM
  Sau khi có preference_dataset.jsonl:

  {
    "prompt": "...",
    "chosen": "...",
    "rejected": "..."
  }

  Có thể dùng:

  DPO
  ORPO
  KTO
  Reward model

  Thực tế nên chọn:

  DPO

  Vì đơn giản hơn RLHF truyền thống, không cần build reward model riêng ngay.

  Luồng:

  SFT trước
  -> model_sft
  -> DPO bằng chosen/rejected
  -> model_dpo
  -> eval
  -> deploy nếu tốt hơn

  Điều Cần Lưu Ý
  Fine-tune local model không tự cập nhật kiến thức pháp luật mới tốt bằng RAG. Vẫn phải giữ RAG.

  Fine-tune chủ yếu cải thiện:

  văn phong
  format trả lời
  cách hỏi lại
  cách từ chối
  cách bám context
  khả năng trả lời ổn định theo mẫu đã duyệt

  Không nên nhồi toàn bộ văn bản luật vào fine-tuning. Văn bản luật nên nằm trong RAG/chunks/vector index.

  Phương Án Tối Ưu Cho Project
  Tôi đề xuất 3 bước:

  Bước 1: Giữ Gemini + hoàn thiện collect/review/export

  Để demo nhanh.

  Bước 2: Thêm local provider chạy Qwen qua Ollama

  Để chứng minh app có thể đổi model.

  Bước 3: Khi có đủ 300-1000 mẫu approved/edited, fine-tune Qwen bằng LoRA

  Sau đó app gọi model fine-tuned qua OpenAI-compatible endpoint.

  Tóm Lại
  Đổi sang Qwen/Llama local thì hoàn toàn khả thi. Project của bạn đã gần sẵn vì đang dùng OpenAI-compatible client. Việc cần làm là:

  1. Thêm LOCAL_LLM_URL và LOCAL_LLM_MODEL
  2. Sửa LLMFactory/model selection
  3. Chạy local model bằng Ollama/vLLM/LM Studio
  4. Export dataset từ Firebase
  5. Fine-tune Qwen/Llama bằng LoRA/QLoRA
  6. Serve model fine-tuned
  7. App gọi endpoint local thay Gemini

  Với tiếng Việt, tôi khuyên bắt đầu bằng:

  Qwen2.5-7B-Instruct + Ollama cho inference demo
  Qwen2.5-7B-Instruct + LoRA bằng LLaMA-Factory/Unsloth cho fine-tuning thật
=================================================================================
Có thể chia nhóm chạy đồng thời như sau:
Task 1: chạy một mình trước vì nó audit kiến trúc và xác định điểm tích hợp.
Sau khi Task 1 xong, Task 2 và Task 3 có thể chạy song song nếu bạn tách ownership file rõ ràng:
Task 2 phụ trách config + Firebase export
Task 3 phụ trách LLM provider + dataset validation/conversion + preference dataset
Sau khi Task 2 và 3 xong, Task 4 và Task 5 có thể chạy song song:
Task 4 phụ trách Local LLM smoke test + SFT/LoRA training package
Task 5 phụ trách Streamlit integration + Base vs SFT evaluation
Sau đó Task 6 và Task 7 có thể chạy song song một phần nếu Task 7 chỉ chuẩn bị evaluation harness trước:
Task 6: RAG regression + DPO training package
Task 7: Base vs SFT vs DPO evaluation
Nhưng phần chạy evaluation thật của Task 7 phải đợi DPO checkpoint từ Task 6.
Task 8 phải đợi Task 7 để biết model nào được chọn deploy.
Task 9 phải chạy cuối cùng sau Task 8 để E2E validation.
============
Task 1
  │
  ├───────────────┐
  ▼               ▼
Task 2          Task 3
  │               │
  └───────┬───────┘
          ▼
   ┌──────────────┐
   ▼              ▼
Task 4          Task 5
   │              │
   └───────┬──────┘
           ▼
    ┌─────────────┐
    ▼             ▼
 Task 6         Task 7*
    │             │
    └──────┬──────┘
           ▼
         Task 8
           │
           ▼
         Task 9