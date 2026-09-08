import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from src.api.schemas import ChatRequest, ChatResponse
from src.rag_flow.context_retriever import ContextRetriever
from src.rag_flow.reasoning_chain import ReasoningChain
from src.llm.prompt_templates import build_prompt
from src.llm.model_factory import LLMFactory
from src.llm.service import LLMService
from src.utils.logger import setup_logger
from configs.settings import settings

app = FastAPI(title="DVC-BCA-RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")

logger = setup_logger()


def _rag_diagnostics(query, contexts):
    """Return non-secret retrieval/provider diagnostics for the chat stream."""
    document_ids = []
    for context in contexts:
        if isinstance(context, dict):
            metadata = context.get("metadata", {})
            metadata = metadata if isinstance(metadata, dict) else {}
            identifier = (
                metadata.get("document_id")
                or metadata.get("source")
                or metadata.get("filename")
                or context.get("document_id")
            )
        else:
            identifier = None
        document_ids.append(identifier or "unknown")
    return {
        "query": query,
        "retrieved_document_ids": document_ids,
        "retrieved_chunks": len(contexts),
        "provider": settings.LLM_PROVIDER,
        "model": LLMFactory.get_model_name(),
    }

# Khởi tạo các thành phần
retriever = ContextRetriever()
reasoning = ReasoningChain()
llm_service = LLMService()

# Load chunks để dùng cho reasoning
chunks = []
try:
    with open(settings.CHUNKS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
except FileNotFoundError:
    logger.warning(f"File {settings.CHUNKS_FILE} không tồn tại. Danh sách chunks đang trống. Vui lòng chạy ingestion_pipeline.py.")
except Exception as e:
    logger.error(f"Lỗi khi load chunks: {e}")

@app.post("/chat")
async def chat(request: ChatRequest):
    query = request.query
    logger.info(f"Query: {query}")

    # 1. Kiểm tra khẩn cấp
    query_lower = query.lower()
    info_keywords = ["quy định", "luật", "thủ tục", "hồ sơ", "pccc", "phòng cháy", "chữa cháy", "hướng dẫn", "mức phạt", "xử phạt", "cấp phép", "giấy phép", "như thế nào", "là gì"]
    is_info_query = any(kw in query_lower for kw in info_keywords)
    
    emergency_keywords = ["cháy", "nổ", "đánh nhau", "đe dọa", "cấp cứu", "tai nạn"]
    is_emergency = any(kw in query_lower for kw in emergency_keywords) and not is_info_query
    
    if is_emergency:
        async def emergency_stream():
            meta = json.dumps({"emergency": True, "need_clarification": False, "contexts": []}, ensure_ascii=False)
            yield f"data: {meta}\n\n"
            ans = "Anh/chị hãy gọi ngay số khẩn cấp: 113 (Công an), 114 (cháy nổ, cứu nạn), 115 (cấp cứu y tế)."
            yield f"data: {json.dumps({'delta': ans}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(emergency_stream(), media_type="text/event-stream")

    # 2. Truy xuất ngữ cảnh
    context_result = retriever.get_context(query)
    contexts = context_result.get("contexts", [])
    entities = context_result.get("entities", {})
    diagnostics = _rag_diagnostics(query, contexts)
    logger.info(
        "RAG retrieval: query=%r provider=%s model=%s chunks=%d document_ids=%s",
        query,
        diagnostics["provider"],
        diagnostics["model"],
        diagnostics["retrieved_chunks"],
        diagnostics["retrieved_document_ids"],
    )

    # 3. Guardrail: Từ chối trả lời nếu không có ngữ cảnh nào khớp (Dẹp TOP-K ép buộc)
    if not contexts:
        async def no_data_stream():
            meta = json.dumps(
                {
                    "emergency": False,
                    "need_clarification": False,
                    "contexts": [],
                    **diagnostics,
                },
                ensure_ascii=False,
            )
            yield f"data: {meta}\n\n"
            ans = "Xin lỗi, dữ liệu hiện tại không đủ cung cấp câu trả lời cho bạn."
            yield f"data: {json.dumps({'delta': ans}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(no_data_stream(), media_type="text/event-stream")

    # 4. Kiểm tra đủ thông tin (Reasoning)
    intent = contexts[0].get('metadata', {}).get('intent_code') if contexts else None
    reasoning_result = reasoning.process(query, intent, entities, chunks)

    if not reasoning_result.get("ready"):
        clar = reasoning_result.get("clarification", "Anh/chị vui lòng cung cấp thêm thông tin cụ thể.")
        async def clarification_stream():
            meta = json.dumps(
                {
                    "emergency": False,
                    "need_clarification": True,
                    "contexts": contexts,
                    "clarification": clar,
                    **diagnostics,
                },
                ensure_ascii=False,
            )
            yield f"data: {meta}\n\n"
            yield f"data: {json.dumps({'delta': clar}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(clarification_stream(), media_type="text/event-stream")

    # 5. Tạo prompt và gọi LLM với Streaming
    # Chỉ dùng những chunk đã vượt qua được Similarity Threshold
    context_text = "\n---\n".join([c.get('text', '') for c in contexts])
    prompt = build_prompt(context_text, query)
    logger.debug("RAG prompt: %s", prompt)

    async def llm_stream():
        meta = json.dumps(
            {
                "emergency": False,
                "need_clarification": False,
                "contexts": contexts,
                **diagnostics,
            },
            ensure_ascii=False,
        )
        yield f"data: {meta}\n\n"
        try:
            full_content = ""
            async for delta in llm_service.stream_chat(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=settings.RESPONSE_MAX_TOKENS,
            ):
                full_content += delta
                yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
            
            logger.info("LLM stream finished successfully.")
            logger.info(
                "RAG answer: query=%r provider=%s answer=%r",
                query,
                diagnostics["provider"],
                full_content,
            )
            if not full_content.strip():
                fallback = "Xin lỗi, hiện tại tôi chưa tìm thấy thông tin chi tiết hoặc câu trả lời chưa sẵn sàng. Anh/chị vui lòng thử lại hoặc liên hệ trực tiếp cơ quan công an."
                yield f"data: {json.dumps({'delta': fallback}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"LLM error: {e}")
            err_msg = f"Lỗi từ máy chủ AI: {e}"
            yield f"data: {json.dumps({'delta': err_msg}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(llm_stream(), media_type="text/event-stream")
