import streamlit as st
import json
import asyncio
import os
import sys

# Thêm đường dẫn gốc để import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_flow.context_retriever import ContextRetriever
from src.rag_flow.reasoning_chain import ReasoningChain
from src.llm.model_factory import LLMFactory
from src.llm.prompt_templates import build_prompt
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
    llm_client = LLMFactory.get_llm()
    chunks = []
    try:
        with open(settings.CHUNKS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line))
    except Exception as e:
        st.warning(f"Lỗi khi load chunks: {e}")
    return retriever, reasoning, llm_client, chunks

retriever, reasoning, llm_client, chunks = load_rag_components_v2()

# Giao diện chính
st.title("🚓 Trợ lý Ảo - Công an xã An Viên")
st.markdown("**Hỗ trợ:** Thủ tục Hành chính, Cư trú, Đăng ký xe, PCCC & An ninh trật tự")

# Khởi tạo session state lưu trữ lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Hiển thị nguồn trích dẫn nếu có
        if "contexts" in msg and msg["contexts"]:
            with st.expander("Nguồn tài liệu trích dẫn", expanded=False):
                for idx, ctx in enumerate(msg["contexts"]):
                    st.markdown(f"**Nguồn {idx + 1}:**\n{ctx.get('text', '')}")

# Khung nhập câu hỏi
if prompt := st.chat_input("Nhập câu hỏi của bạn tại đây..."):
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
            st.session_state.messages.append({"role": "assistant", "content": ans})
            st.stop()

        # 2. Truy xuất ngữ cảnh
        with st.spinner("Đang tìm kiếm thông tin..."):
            context_result = retriever.get_context(prompt)
            contexts = context_result.get("contexts", [])
            entities = context_result.get("entities", {})

        if not contexts:
            ans = "Xin lỗi, dữ liệu hiện tại không đủ cung cấp câu trả lời cho bạn."
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
            st.stop()

        # 3. Reasoning (Kiểm tra đủ thông tin)
        intent = contexts[0].get('metadata', {}).get('intent_code') if contexts else None
        reasoning_result = reasoning.process(prompt, intent, entities, chunks)

        if not reasoning_result.get("ready"):
            clar = reasoning_result.get("clarification", "Anh/chị vui lòng cung cấp thêm thông tin cụ thể.")
            st.markdown(clar)
            
            with st.expander("Nguồn tài liệu trích dẫn", expanded=False):
                for idx, ctx in enumerate(contexts):
                    st.markdown(f"**Nguồn {idx + 1}:**\n{ctx.get('text', '')}")
                    
            st.session_state.messages.append({"role": "assistant", "content": clar, "contexts": contexts})
            st.stop()

        # 4. LLM Generation
        context_text = "\n---\n".join([c.get('text', '') for c in contexts])
        prompt_text = build_prompt(context_text, prompt)
        
        message_placeholder = st.empty()
        
        async def fetch_llm_stream():
            model_name = settings.OPENAI_MODEL if settings.LLM_PROVIDER == "openai" else settings.GEMINI_MODEL
            stream = await llm_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.3,
                max_tokens=2048,
                stream=True
            )
            full_response = ""
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    full_response += delta
                    message_placeholder.markdown(full_response + "▌")
            return full_response

        try:
            full_response = asyncio.run(fetch_llm_stream())
            message_placeholder.markdown(full_response)
            
            with st.expander("Nguồn tài liệu trích dẫn", expanded=False):
                for idx, ctx in enumerate(contexts):
                    st.markdown(f"**Nguồn {idx + 1}:**\n{ctx.get('text', '')}")
                    
            st.session_state.messages.append({"role": "assistant", "content": full_response, "contexts": contexts})
        except Exception as e:
            st.error(f"Lỗi từ máy chủ AI: {e}")
