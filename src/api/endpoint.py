import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import FastAPI, HTTPException
from src.api.schemas import ChatRequest, ChatResponse
from src.rag_flow.context_retriever import ContextRetriever
from src.rag_flow.reasoning_chain import ReasoningChain
from src.llm.model_factory import LLMFactory
from src.llm.prompt_templates import build_prompt
from src.utils.logger import setup_logger
from configs.settings import settings

app = FastAPI(title="DVC-BCA-RAG Chatbot")
logger = setup_logger()

# Khởi tạo các thành phần
retriever = ContextRetriever()
reasoning = ReasoningChain()
llm_client = LLMFactory.get_llm()

# Load chunks để dùng cho reasoning (có thể từ file chunks.jsonl)
import json
chunks = []
with open(settings.CHUNKS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        chunks.append(json.loads(line))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    query = request.query
    logger.info(f"Query: {query}")

    # 1. Kiểm tra khẩn cấp
    emergency_keywords = ["cháy", "nổ", "đánh nhau", "đe dọa", "cấp cứu", "tai nạn"]
    if any(kw in query.lower() for kw in emergency_keywords):
        return ChatResponse(
            answer="Anh/chị hãy gọi ngay số khẩn cấp: 113 (Công an), 114 (cháy nổ, cứu nạn), 115 (cấp cứu y tế).",
            emergency=True,
            need_clarification=False
        )

    # 2. Truy xuất ngữ cảnh
    context_result = retriever.get_context(query)
    contexts = context_result.get("contexts", [])
    entities = context_result.get("entities", {})

    # 3. Kiểm tra đủ thông tin
    intent = None  # có thể xác định từ chunk có điểm số cao nhất
    if contexts:
        intent = contexts[0].get('metadata', {}).get('intent_code')  # giả định có trường này
    reasoning_result = reasoning.process(query, intent, entities, chunks)

    if not reasoning_result.get("ready"):
        return ChatResponse(
            answer="",
            need_clarification=True,
            clarification_question=reasoning_result.get("clarification", "Anh/chị vui lòng cung cấp thêm thông tin cụ thể.")
        )

    # 4. Tạo prompt và gọi LLM
    context_text = "\n---\n".join([c.get('text', '') for c in contexts[:3]])  # chỉ lấy 3 chunk đầu
    prompt = build_prompt(context_text, query)

    try:
        response = llm_client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        answer = response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM error: {e}")
        # Fallback: dùng câu trả lời từ chunk nếu có
        if contexts:
            answer = contexts[0].get('text', "Xin lỗi, hiện tại tôi chưa thể xử lý yêu cầu của anh/chị.")
        else:
            answer = "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp. Anh/chị vui lòng liên hệ trực tiếp Công an phường."

    return ChatResponse(
        answer=answer,
        context_used=contexts,
        need_clarification=False
    )