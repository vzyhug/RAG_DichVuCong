import streamlit as st
import json
import asyncio
import os
import sys
import re
from uuid import uuid4

# Thêm đường dẫn gốc để import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_flow.context_retriever import ContextRetriever
from src.rag_flow.reasoning_chain import ReasoningChain
from src.llm.model_factory import LLMFactory
from src.llm.prompt_templates import build_prompt
from src.llm.service import LLMService
from src.training_data.collector import collect_chat_log
from src.training_data.firebase_store import (
    list_chat_logs,
    update_chat_feedback,
    update_review_status,
)
from configs.settings import settings

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Công an xã An Viên - Trợ lý ảo",
    page_icon="🚓",
    layout="centered"
)

# Khởi tạo các thành phần AI (Cache để không phải load lại mỗi lần render)
@st.cache_resource
def load_rag_components_v2():
    retriever = ContextRetriever()
    reasoning = ReasoningChain()
    llm_service = LLMService()
    chunks = []
    try:
        with open(settings.CHUNKS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line))
    except Exception as e:
        st.warning(f"Lỗi khi load chunks: {e}")
    return retriever, reasoning, llm_service, chunks


def _context_filename(context):
    metadata = context.get("metadata", {}) if isinstance(context, dict) else {}
    if not isinstance(metadata, dict):
        return "unknown"
    return metadata.get("filename") or metadata.get("source_id") or "unknown"


def render_review_data():
    """Render the internal review queue backed by Firestore."""
    st.title("Review data")
    filter_columns = st.columns(2)
    with filter_columns[0]:
        status_filter = st.selectbox(
            "Review status",
            ["All", "raw", "approved", "rejected", "edited"],
            key="review_status_filter",
        )
    with filter_columns[1]:
        rating_filter = st.selectbox(
            "Rating",
            ["All", "up", "down", "wrong"],
            key="review_rating_filter",
        )

    logs = list_chat_logs(
        limit=100,
        review_status=None if status_filter == "All" else status_filter,
        rating=None if rating_filter == "All" else rating_filter,
    )
    if not logs:
        st.info("No chat logs found.")
        return

    for index, log in enumerate(logs):
        turn_id = str(log.get("turn_id") or log.get("document_id") or index)
        current_status = log.get("review_status") or "raw"
        with st.expander(
            f"{log.get('user_query', 'Untitled query')} [{current_status}]",
            expanded=False,
        ):
            st.markdown("**User query**")
            st.write(log.get("user_query", ""))
            st.markdown("**Assistant answer**")
            st.write(log.get("assistant_answer", ""))
            st.markdown("**Response type**")
            st.write(log.get("response_type", ""))
            st.markdown("**Feedback**")
            st.write(log.get("feedback") or "No feedback")

            st.markdown("**Contexts**")
            contexts = log.get("contexts") or []
            if contexts:
                st.table(
                    [
                        {
                            "filename": _context_filename(context),
                            "score": context.get("score") if isinstance(context, dict) else None,
                        }
                        for context in contexts
                    ]
                )
            else:
                st.caption("No contexts")

            corrected_answer = st.text_area(
                "Corrected answer",
                value=log.get("corrected_answer") or "",
                key=f"corrected_answer_{turn_id}",
            )
            error_type = st.text_input(
                "Error type",
                value=log.get("error_type") or "",
                key=f"error_type_{turn_id}",
            )
            selected_status = st.selectbox(
                "Review status",
                ["raw", "approved", "rejected", "edited"],
                index=(
                    ["raw", "approved", "rejected", "edited"].index(current_status)
                    if current_status in {"raw", "approved", "rejected", "edited"}
                    else 0
                ),
                key=f"review_status_{turn_id}",
            )
            if st.button("Save review", key=f"save_review_{turn_id}"):
                if update_review_status(
                    turn_id=turn_id,
                    review_status=selected_status,
                    corrected_answer=corrected_answer or None,
                    error_type=error_type or None,
                ):
                    st.success("Review saved.")
                    st.rerun()
                else:
                    st.error("Could not save the review.")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = set()

page = st.sidebar.selectbox("View", ["Chatbot", "Review data"], key="app_page")
if page == "Review data":
    render_review_data()
    st.stop()

retriever, reasoning, llm_service, chunks = load_rag_components_v2()

# Giao diện chính
st.title("🚓 Trợ lý Ảo - Công an xã An Viên")
st.markdown("**Hỗ trợ:** Thủ tục Hành chính, Cư trú, Đăng ký xe, PCCC & An ninh trật tự")


def log_response(
    user_query,
    turn_id,
    answer,
    response_type,
    contexts=None,
    entities=None,
):
    document_id = collect_chat_log(
        user_query=user_query,
        assistant_answer=answer,
        response_type=response_type,
        contexts=contexts,
        entities=entities,
        model=LLMFactory.get_model_name(),
        provider=settings.LLM_PROVIDER,
        session_id=st.session_state.session_id,
        turn_id=turn_id,
    )
    if document_id:
        st.session_state.last_logged_turn_id = turn_id


def render_feedback(turn_id):
    """Render one-time feedback controls for an assistant turn."""
    if not turn_id:
        return

    if turn_id in st.session_state.feedback_submitted:
        st.caption("Feedback submitted")
        return

    st.caption("Rate this answer")
    feedback_options = (("Helpful", "up"), ("Not helpful", "down"), ("Incorrect information", "wrong"))
    feedback_columns = st.columns(len(feedback_options))
    for column, (label, rating) in zip(feedback_columns, feedback_options):
        with column:
            if st.button(label, key=f"feedback_{rating}_{turn_id}"):
                if update_chat_feedback(turn_id, {"rating": rating}):
                    st.session_state.feedback_submitted.add(turn_id)
                    st.rerun()
                else:
                    st.error("Could not submit feedback.")

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Hiển thị nguồn trích dẫn nếu có
        if "contexts" in msg and msg["contexts"]:
            with st.expander("Nguồn tài liệu trích dẫn", expanded=False):
                st.markdown("Nguồn trích dẫn từ tài liệu nội bộ...")
        if msg["role"] == "assistant":
            render_feedback(msg.get("turn_id"))

# Khung nhập câu hỏi
if prompt := st.chat_input("Nhập câu hỏi của bạn tại đây..."):
    turn_id = str(uuid4())
    # Thêm câu hỏi vào lịch sử và hiển thị
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. Kiểm tra khẩn cấp
        query_lower = prompt.lower()
        info_keywords = ["quy định", "luật", "thủ tục", "hồ sơ", "pccc", "phòng cháy", "chữa cháy", "hướng dẫn", "mức phạt", "xử phạt", "cấp phép", "giấy phép", "như thế nào", "là gì"]
        is_info_query = any(kw in query_lower for kw in info_keywords)
        
        emergency_keywords = ["cháy", "nổ", "đánh nhau", "đe dọa", "cấp cứu", "tai nạn", "trộm cắp", "cướp", "giết người"]
        is_emergency = any(kw in query_lower for kw in emergency_keywords) and not is_info_query
        
        if is_emergency:
            ans = "🚨 **TÌNH HUỐNG KHẨN CẤP / BÁO TIN TỘI PHẠM**\n\nAnh/chị vui lòng liên hệ ngay lập tức:\n- **Trực ban Công an xã An Viên:** [Chèn SĐT trực ban vào đây]\n- **Cảnh sát 113** (An ninh trật tự)\n- **Cứu hỏa 114** (Cháy nổ, cứu nạn)\n- **Cấp cứu 115** (Y tế)"
            st.warning(ans)
            log_response(prompt, turn_id, ans, "emergency", contexts=[], entities={})
            st.session_state.messages.append({"role": "assistant", "content": ans, "turn_id": turn_id})
            render_feedback(turn_id)
            st.stop()

        # 2. Truy xuất ngữ cảnh
        with st.spinner("Đang tìm kiếm thông tin..."):
            context_result = retriever.get_context(prompt)
            contexts = context_result.get("contexts", [])
            entities = context_result.get("entities", {})

        if not contexts:
            ans = "Xin lỗi, dữ liệu hiện tại không đủ cung cấp câu trả lời cho bạn."
            st.markdown(ans)
            log_response(prompt, turn_id, ans, "no_data", contexts=contexts, entities=entities)
            st.session_state.messages.append({"role": "assistant", "content": ans, "turn_id": turn_id})
            render_feedback(turn_id)
            st.stop()

        # 3. Reasoning (Kiểm tra đủ thông tin)
        intent = contexts[0].get('metadata', {}).get('intent_code') if contexts else None
        reasoning_result = reasoning.process(prompt, intent, entities, chunks)

        if not reasoning_result.get("ready"):
            clar = reasoning_result.get("clarification", "Anh/chị vui lòng cung cấp thêm thông tin cụ thể.")
            st.markdown(clar)
            
            with st.expander("Nguồn tài liệu trích dẫn", expanded=False):
                st.markdown("Nguồn trích dẫn từ tài liệu nội bộ...")
                    
            st.session_state.messages.append(
                {"role": "assistant", "content": clar, "contexts": contexts, "turn_id": turn_id}
            )
            log_response(prompt, turn_id, clar, "clarification", contexts=contexts, entities=entities)
            render_feedback(turn_id)
            st.stop()

        # 4. LLM Generation
        context_text = "\n---\n".join([re.sub(r'\.{3,}', '', c.get('text', '')) for c in contexts])
        prompt_text = build_prompt(context_text, prompt)
        
        message_placeholder = st.empty()
        
        async def fetch_llm_stream():
            # Lọc lấy lịch sử chat (loại bỏ trường 'contexts' và câu hỏi hiện tại)
            history_messages = [
                {"role": msg["role"], "content": msg["content"]} 
                for msg in st.session_state.messages[:-1]
            ]
            # Nối lịch sử với câu hỏi hiện tại (đã kèm ngữ cảnh tài liệu)
            messages_for_llm = history_messages + [{"role": "user", "content": prompt_text}]

            full_response = ""
            async for delta in llm_service.stream_chat(
                messages_for_llm,
                temperature=0.3,
                max_tokens=settings.RESPONSE_MAX_TOKENS,
            ):
                full_response += delta
                message_placeholder.markdown(full_response + "▌")
            return full_response

        try:
            full_response = asyncio.run(fetch_llm_stream())
            message_placeholder.markdown(full_response)
            
            with st.expander("Nguồn tài liệu trích dẫn", expanded=False):
                st.markdown("Nguồn trích dẫn từ tài liệu nội bộ...")
                    
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "contexts": contexts,
                    "turn_id": turn_id,
                }
            )
            log_response(prompt, turn_id, full_response, "normal", contexts=contexts, entities=entities)
            render_feedback(turn_id)
        except Exception as e:
            error_answer = f"Lỗi từ máy chủ AI: {e}"
            st.error(error_answer)
            log_response(prompt, turn_id, error_answer, "error", contexts=contexts, entities=entities)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_answer, "turn_id": turn_id}
            )
            render_feedback(turn_id)
