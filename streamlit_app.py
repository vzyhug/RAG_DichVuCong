import streamlit as st
import json
import asyncio
import os
import sys
import re
import html
from uuid import uuid4

# Thêm đường dẫn gốc để import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_flow.context_retriever import ContextRetriever
from src.rag_flow.reasoning_chain import ReasoningChain
from src.llm.model_factory import LLMFactory
from src.llm.prompt_templates import build_prompt
from src.training_data.collector import collect_chat_log
from src.training_data.firebase_store import (
    create_data_update_request,
    list_chat_logs,
    update_chat_feedback,
    update_review_status,
)
from src.training_data.github_issues import (
    create_github_issue,
    is_github_issue_enabled,
    is_github_project_enabled,
)
from configs.settings import settings

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Công an xã An Viễn - Trợ lý ảo",
    page_icon="🚓",
    layout="wide"
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


def _context_filename(context):
    metadata = context.get("metadata", {}) if isinstance(context, dict) else {}
    if not isinstance(metadata, dict):
        return "unknown"
    return metadata.get("filename") or metadata.get("source_id") or "unknown"


def _feedback_rating(log):
    feedback = log.get("feedback") or {}
    if isinstance(feedback, dict):
        return feedback.get("rating") or "none"
    return "none"


def _status_label(status):
    labels = {
        "All": "Tất cả",
        "raw": "Chưa review",
        "approved": "Đã duyệt",
        "rejected": "Từ chối",
        "edited": "Đã chỉnh sửa",
    }
    return labels.get(status, status)


def _rating_label(rating):
    labels = {
        "All": "Tất cả",
        "up": "Hữu ích",
        "down": "Chưa hữu ích",
        "wrong": "Sai thông tin",
        "none": "Chưa có",
    }
    return labels.get(rating, rating)


def _response_type_label(response_type):
    labels = {
        "normal": "Trả lời thường",
        "no_data": "Không có dữ liệu",
        "emergency": "Khẩn cấp",
        "unknown": "Không rõ",
    }
    return labels.get(response_type, response_type)


def _data_issue_label(issue_type):
    labels = {
        "missing_data": "Thiếu dữ liệu để trả lời",
        "wrong_data": "Dữ liệu đang sai",
        "outdated_data": "Thông tin đã cũ",
        "incomplete_answer": "Câu trả lời chưa đủ ý",
        "wrong_source": "Bot dùng sai nguồn",
        "other": "Khác",
    }
    return labels.get(issue_type, issue_type)


def _short_text(value, length=92):
    value = (value or "").replace("\n", " ").strip()
    if len(value) <= length:
        return value or "Câu hỏi chưa có tiêu đề"
    return f"{value[:length].rstrip()}..."


def _escape_html(value):
    return html.escape(str(value or ""))


def _render_review_css():
    st.markdown(
        """
        <style>
        section.main > div {
            padding-top: 1.5rem;
        }
        .review-hero {
            border: 1px solid rgba(49, 51, 63, 0.12);
            border-radius: 10px;
            padding: 18px 20px;
            margin-bottom: 16px;
            background: linear-gradient(135deg, #f8fafc 0%, #eef6ff 100%);
        }
        .review-hero h1 {
            margin: 0 0 6px 0;
            font-size: 1.8rem;
            line-height: 1.2;
            color: #102033;
        }
        .review-hero p {
            margin: 0;
            color: #526173;
            font-size: 0.98rem;
        }
        .review-card-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 10px;
            margin: 14px 0 18px 0;
        }
        .review-kpi-card {
            border: 1px solid rgba(49, 51, 63, 0.12);
            border-radius: 8px;
            padding: 12px 14px;
            background: #ffffff;
            min-height: 84px;
        }
        .review-kpi-label {
            font-size: 0.82rem;
            color: #667085;
            margin-bottom: 8px;
        }
        .review-kpi-value {
            font-size: 1.6rem;
            font-weight: 750;
            color: #111827;
            line-height: 1;
        }
        .review-kpi-accent {
            width: 34px;
            height: 3px;
            border-radius: 999px;
            background: #2563eb;
            margin-top: 12px;
        }
        .record-shell {
            border: 1px solid rgba(49, 51, 63, 0.14);
            border-radius: 10px;
            background: #ffffff;
            padding: 16px 18px;
            margin: 12px 0;
        }
        .record-meta {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: 10px 0 12px 0;
        }
        .record-meta-item {
            border: 1px solid rgba(49, 51, 63, 0.10);
            border-radius: 8px;
            padding: 10px 12px;
            background: #f9fafb;
        }
        .record-meta-label {
            color: #667085;
            font-size: 0.78rem;
            margin-bottom: 4px;
        }
        .record-meta-value {
            color: #111827;
            font-weight: 700;
            font-size: 0.95rem;
        }
        .review-hint {
            border-left: 4px solid #2563eb;
            background: #eff6ff;
            color: #1e3a8a;
            padding: 10px 12px;
            border-radius: 8px;
            margin: 10px 0 14px 0;
        }
        .chat-panel {
            border: 1px solid rgba(49, 51, 63, 0.12);
            border-radius: 10px;
            padding: 14px 16px;
            margin: 12px 0;
            background: #ffffff;
        }
        .chat-panel.question {
            background: #f8fafc;
        }
        .chat-panel.answer {
            background: #fffdf7;
            border-color: rgba(217, 119, 6, 0.20);
        }
        .chat-panel-title {
            color: #475467;
            font-weight: 700;
            font-size: 0.9rem;
            margin-bottom: 8px;
        }
        .chat-panel-body {
            color: #111827;
            font-size: 1rem;
            line-height: 1.65;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
        }
        .review-section-title {
            font-size: 1.05rem;
            font-weight: 750;
            color: #111827;
            margin: 16px 0 8px 0;
        }
        .st-key-sticky_review_actions {
            position: sticky;
            top: 0;
            z-index: 1000;
            background: rgba(255, 255, 255, 0.97);
            border: 1px solid rgba(49, 51, 63, 0.16);
            border-radius: 10px;
            padding: 0.55rem 0.75rem;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.11);
            backdrop-filter: blur(8px);
            margin: 0.75rem 0;
        }
        .st-key-sticky_review_actions [data-testid="stCaptionContainer"] {
            color: #475467;
            font-weight: 700;
        }
        @media (max-width: 900px) {
            .review-card-grid,
            .record-meta {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (prefers-color-scheme: dark) {
            .review-hero,
            .review-kpi-card,
            .record-shell,
            .record-meta-item,
            .chat-panel {
                background: #111827;
                border-color: rgba(255, 255, 255, 0.14);
            }
            .review-hero h1,
            .review-kpi-value,
            .record-meta-value,
            .chat-panel-body,
            .review-section-title {
                color: #f9fafb;
            }
            .review-hero p,
            .review-kpi-label,
            .record-meta-label,
            .chat-panel-title {
                color: #cbd5e1;
            }
            .chat-panel.question {
                background: #0f172a;
            }
            .chat-panel.answer {
                background: #1f2937;
            }
            .review-hint {
                background: #172554;
                color: #dbeafe;
            }
            .st-key-sticky_review_actions {
                background: rgba(17, 24, 39, 0.97);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_review_hero():
    st.markdown(
        """
        <div class="review-hero">
            <h1>Dữ liệu review</h1>
            <p>Duyệt câu hỏi và câu trả lời đã thu thập để cải thiện chất lượng trả lời của chatbot.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_kpi_cards(log_count, status_counts):
    cards = [
        ("Tổng", log_count),
        ("Chưa review", status_counts.get("raw", 0)),
        ("Đã duyệt", status_counts.get("approved", 0)),
        ("Đã chỉnh sửa", status_counts.get("edited", 0)),
        ("Từ chối", status_counts.get("rejected", 0)),
    ]
    html_cards = "".join(
        f"""
        <div class="review-kpi-card">
            <div class="review-kpi-label">{_escape_html(label)}</div>
            <div class="review-kpi-value">{value}</div>
            <div class="review-kpi-accent"></div>
        </div>
        """
        for label, value in cards
    )
    st.markdown(f'<div class="review-card-grid">{html_cards}</div>', unsafe_allow_html=True)


def _render_record_meta(current_status, rating, response_type, contexts_count):
    items = [
        ("Trạng thái", _status_label(current_status)),
        ("Đánh giá", _rating_label(rating)),
        ("Loại phản hồi", _response_type_label(response_type)),
        ("Ngữ cảnh", str(contexts_count)),
    ]
    html_items = "".join(
        f"""
        <div class="record-meta-item">
            <div class="record-meta-label">{_escape_html(label)}</div>
            <div class="record-meta-value">{_escape_html(value)}</div>
        </div>
        """
        for label, value in items
    )
    st.markdown(f'<div class="record-meta">{html_items}</div>', unsafe_allow_html=True)


def _render_hint(message):
    st.markdown(
        f'<div class="review-hint">{_escape_html(message)}</div>',
        unsafe_allow_html=True,
    )


def _render_chat_panel(title, body, panel_type):
    st.markdown(
        f"""
        <div class="chat-panel {panel_type}">
            <div class="chat-panel-title">{_escape_html(title)}</div>
            <div class="chat-panel-body">{_escape_html(body) or "_Chưa có nội dung_"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _review_action_hint(log):
    status = log.get("review_status") or "raw"
    error_type = log.get("error_type") or ""
    response_type = log.get("response_type") or ""

    if status == "approved":
        return "Câu trả lời đã ổn, có thể dùng làm mẫu tham khảo."
    if status == "edited":
        return "Đã có bản chỉnh sửa. Nếu lỗi do thiếu hoặc sai dữ liệu, hãy gửi đội kỹ thuật xử lý."
    if status == "rejected":
        return "Câu trả lời bị loại. Chỉ gửi đội kỹ thuật nếu vấn đề nằm ở dữ liệu nguồn."
    if error_type in {"missing_context", "no_data"} or response_type == "no_data":
        return "Có khả năng hệ thống đang thiếu dữ liệu phù hợp để trả lời."
    if error_type == "wrong_context":
        return "Có khả năng chatbot dùng sai nguồn thông tin."
    return "Đọc câu hỏi, câu trả lời và nguồn tham khảo rồi chọn trạng thái phù hợp."


def _count_by(logs, field, default="none"):
    counts = {}
    for log in logs:
        value = log.get(field) or default
        counts[value] = counts.get(value, 0) + 1
    return counts


def _review_status_index(status):
    statuses = ["raw", "approved", "rejected", "edited"]
    return statuses.index(status) if status in statuses else 0


def _context_rows(contexts):
    rows = []
    for context in contexts:
        metadata = context.get("metadata", {}) if isinstance(context, dict) else {}
        rows.append(
            {
                "filename": _context_filename(context),
                "score": context.get("score") if isinstance(context, dict) else None,
                "intent": metadata.get("intent_code") if isinstance(metadata, dict) else None,
            }
        )
    return rows


def _build_data_update_payload(
    log,
    turn_id,
    selected_status,
    corrected_answer,
    issue_type,
    request_note,
):
    return {
        "source_turn_id": turn_id,
        "source_document_id": log.get("document_id"),
        "status": "open",
        "priority": "normal",
        "issue_type": issue_type,
        "issue_label": _data_issue_label(issue_type),
        "request_note": request_note,
        "user_query": log.get("user_query") or "",
        "assistant_answer": log.get("assistant_answer") or "",
        "corrected_answer": corrected_answer or "",
        "review_status": selected_status,
        "response_type": log.get("response_type") or "",
        "contexts": log.get("contexts") or [],
        "feedback": log.get("feedback"),
        "provider": log.get("provider"),
        "model": log.get("model"),
        "session_id": log.get("session_id"),
    }


def render_review_data():
    """Render the internal review queue backed by Firestore."""
    _render_review_css()
    _render_review_hero()

    filter_columns = st.columns([1, 1, 2, 1])
    with filter_columns[0]:
        status_filter = st.selectbox(
            "Trạng thái review",
            ["All", "raw", "approved", "rejected", "edited"],
            format_func=_status_label,
            key="review_status_filter",
        )
    with filter_columns[1]:
        rating_filter = st.selectbox(
            "Đánh giá",
            ["All", "up", "down", "wrong"],
            format_func=_rating_label,
            key="review_rating_filter",
        )
    with filter_columns[2]:
        search_text = st.text_input(
            "Tìm kiếm",
            placeholder="Tìm theo câu hỏi, câu trả lời hoặc tên file nguồn",
            key="review_search_filter",
        )
    with filter_columns[3]:
        limit = st.number_input(
            "Số bản ghi",
            min_value=20,
            max_value=300,
            value=100,
            step=20,
            key="review_limit",
        )

    logs = list_chat_logs(
        limit=int(limit),
        review_status=None if status_filter == "All" else status_filter,
        rating=None if rating_filter == "All" else rating_filter,
    )

    if search_text:
        keyword = search_text.strip().lower()
        logs = [
            log
            for log in logs
            if keyword in (log.get("user_query") or "").lower()
            or keyword in (log.get("assistant_answer") or "").lower()
            or keyword in " ".join(_context_filename(context) for context in log.get("contexts") or []).lower()
        ]

    if not logs:
        st.info("Không tìm thấy dữ liệu chat.")
        return

    status_counts = _count_by(logs, "review_status", default="raw")
    response_counts = _count_by(logs, "response_type", default="unknown")
    rating_counts = {}
    for log in logs:
        rating = _feedback_rating(log)
        rating_counts[rating] = rating_counts.get(rating, 0) + 1

    _render_kpi_cards(len(logs), status_counts)

    st.divider()
    st.subheader("Không gian review")

    if "review_current_index" not in st.session_state:
        st.session_state.review_current_index = 0
    if st.session_state.review_current_index >= len(logs):
        st.session_state.review_current_index = max(len(logs) - 1, 0)

    nav_columns = st.columns([1, 1, 2, 1, 1])
    with nav_columns[0]:
        if st.button("Trước", use_container_width=True, disabled=st.session_state.review_current_index <= 0):
            st.session_state.review_current_index -= 1
            st.rerun()
    with nav_columns[1]:
        if st.button("Tiếp", use_container_width=True, disabled=st.session_state.review_current_index >= len(logs) - 1):
            st.session_state.review_current_index += 1
            st.rerun()
    with nav_columns[2]:
        st.progress((st.session_state.review_current_index + 1) / len(logs))
        st.caption(f"Bản ghi {st.session_state.review_current_index + 1} / {len(logs)}")
    with nav_columns[3]:
        if st.button("Làm mới", use_container_width=True):
            st.rerun()
    with nav_columns[4]:
        if st.button("Về đầu", use_container_width=True):
            st.session_state.review_current_index = 0
            st.rerun()

    selected_index = st.session_state.review_current_index
    log = logs[selected_index]
    turn_id = str(log.get("turn_id") or log.get("document_id") or selected_index)
    current_status = log.get("review_status") or "raw"
    rating = _feedback_rating(log)
    response_type = log.get("response_type") or "unknown"
    contexts = log.get("contexts") or []

    _render_record_meta(current_status, rating, response_type, len(contexts))
    _render_hint(_review_action_hint(log))

    issue_types = [
        "missing_data",
        "wrong_data",
        "outdated_data",
        "incomplete_answer",
        "wrong_source",
        "other",
    ]
    corrected_key = f"corrected_answer_{turn_id}"
    issue_type_key = f"issue_type_{turn_id}"
    status_key = f"review_status_{turn_id}"
    needs_update_key = f"needs_data_update_{turn_id}"
    note_key = f"data_update_note_{turn_id}"

    st.session_state.setdefault(corrected_key, log.get("corrected_answer") or "")
    st.session_state.setdefault(
        issue_type_key,
        log.get("error_type") if log.get("error_type") in issue_types else "missing_data",
    )
    st.session_state.setdefault(status_key, current_status)
    st.session_state.setdefault(needs_update_key, bool(log.get("needs_data_update")))
    st.session_state.setdefault(note_key, log.get("data_update_note") or "")

    def submit_review(status_to_save=None, force_data_update=None):
        final_status = status_to_save or st.session_state.get(status_key, current_status)
        final_needs_data_update = (
            st.session_state.get(needs_update_key, False)
            if force_data_update is None
            else force_data_update
        )
        final_corrected_answer = st.session_state.get(corrected_key, "")
        final_issue_type = st.session_state.get(issue_type_key, "missing_data")
        final_request_note = st.session_state.get(note_key, "")
        request_id = log.get("data_update_request_id")
        github_issue_url = log.get("github_issue_url")
        github_issue_number = log.get("github_issue_number")
        github_project_item_id = log.get("github_project_item_id")
        github_project_error = None
        github_error = None

        if final_needs_data_update and not github_issue_url:
            request_payload = _build_data_update_payload(
                log=log,
                turn_id=turn_id,
                selected_status=final_status,
                corrected_answer=final_corrected_answer,
                issue_type=final_issue_type,
                request_note=final_request_note,
            )
            if is_github_issue_enabled():
                issue_result = create_github_issue(request_payload)
                if issue_result.get("ok"):
                    github_issue_url = issue_result.get("url")
                    github_issue_number = issue_result.get("number")
                    request_payload["github_issue_url"] = github_issue_url
                    request_payload["github_issue_number"] = github_issue_number
                    github_project_item_id = issue_result.get("project_item_id")
                    github_project_error = issue_result.get("project_error")
                    if github_project_item_id:
                        request_payload["github_project_item_id"] = github_project_item_id
                    if github_project_error:
                        request_payload["github_project_error"] = github_project_error
                else:
                    github_error = issue_result.get("error") or "Không thể tạo GitHub Issue."
                    request_payload["github_error"] = github_error
            else:
                github_error = "GitHub chưa được cấu hình."
                request_payload["github_error"] = github_error

            if not request_id:
                request_id = create_data_update_request(request_payload)

        if update_review_status(
            turn_id=turn_id,
            review_status=final_status,
            corrected_answer=final_corrected_answer or None,
            error_type=final_issue_type if final_needs_data_update else None,
            needs_data_update=final_needs_data_update,
            data_update_note=final_request_note if final_needs_data_update else None,
            data_update_request_id=request_id,
            github_issue_url=github_issue_url,
            github_issue_number=github_issue_number,
            github_project_item_id=github_project_item_id,
            github_project_error=github_project_error,
        ):
            if final_needs_data_update and github_issue_url and github_project_error:
                st.warning(
                    f"Da luu review va tao GitHub Issue, nhung chua dua duoc vao Project: {github_project_error}"
                )
            elif final_needs_data_update and github_issue_url:
                st.success(f"Đã lưu review và gửi yêu cầu cho đội kỹ thuật: {github_issue_url}")
            elif final_needs_data_update and github_error:
                st.warning(f"Đã lưu review, nhưng chưa gửi được GitHub: {github_error}")
            else:
                st.success("Đã lưu review.")
            if st.session_state.review_current_index < len(logs) - 1:
                st.session_state.review_current_index += 1
            st.rerun()
        else:
            st.error("Không thể lưu review.")

    with st.container(key="sticky_review_actions"):
        st.caption("Thao tác nhanh")
        action_columns = st.columns([1, 1, 1.2, 1, 1])
        with action_columns[0]:
            if st.button("Đúng, lưu & tiếp", key=f"sticky_approve_{turn_id}", use_container_width=True):
                submit_review(status_to_save="approved", force_data_update=False)
        with action_columns[1]:
            if st.button("Cần sửa", key=f"sticky_edit_{turn_id}", use_container_width=True):
                submit_review(status_to_save="edited")
        with action_columns[2]:
            if st.button("Gửi đội kỹ thuật", key=f"sticky_send_tech_{turn_id}", use_container_width=True):
                submit_review(status_to_save="edited", force_data_update=True)
        with action_columns[3]:
            if st.button("Lưu lựa chọn", key=f"sticky_save_{turn_id}", use_container_width=True):
                submit_review()
        with action_columns[4]:
            if st.button("Bỏ qua", key=f"sticky_skip_{turn_id}", use_container_width=True):
                if st.session_state.review_current_index < len(logs) - 1:
                    st.session_state.review_current_index += 1
                    st.rerun()

    _render_chat_panel("Câu hỏi của người dùng", log.get("user_query", ""), "question")
    _render_chat_panel("Câu trả lời của chatbot", log.get("assistant_answer", ""), "answer")

    st.markdown('<div class="review-section-title">Thông tin review</div>', unsafe_allow_html=True)
    corrected_answer = st.text_area(
        "Câu trả lời đã chỉnh sửa",
        key=f"corrected_answer_{turn_id}",
        height=180,
    )

    form_columns = st.columns([1, 1, 1.2])
    with form_columns[0]:
        issue_type = st.selectbox(
            "Lý do cần xử lý",
            issue_types,
            format_func=_data_issue_label,
            key=f"issue_type_{turn_id}",
        )
    with form_columns[1]:
        selected_status = st.selectbox(
            "Trạng thái review",
            ["raw", "approved", "rejected", "edited"],
            format_func=_status_label,
            key=f"review_status_{turn_id}",
        )
    with form_columns[2]:
        st.write("")
        st.write("")
        needs_data_update = st.checkbox(
            "Gửi đội kỹ thuật xử lý dữ liệu",
            key=f"needs_data_update_{turn_id}",
        )

    request_note = ""
    if needs_data_update:
        request_note = st.text_area(
            "Ghi chú cho đội kỹ thuật",
            placeholder="Ví dụ: cần bổ sung quy định mới, bot lấy sai nguồn, câu trả lời thiếu bước...",
            key=f"data_update_note_{turn_id}",
            height=100,
        )
        if log.get("github_issue_url"):
            st.info(f"Yêu cầu đã được gửi: {log.get('github_issue_url')}")
            if log.get("github_project_item_id"):
                st.caption("Yêu cầu đã được đưa vào GitHub Project.")
            elif is_github_project_enabled() and log.get("github_project_error"):
                st.warning(f"Chưa đưa được yêu cầu vào GitHub Project: {log.get('github_project_error')}")
            elif is_github_issue_enabled() and not is_github_project_enabled():
                st.caption("GitHub Project chưa cấu hình; yêu cầu vẫn đã được gửi bằng GitHub Issue.")
        elif not is_github_issue_enabled():
            st.warning("Chưa cấu hình GitHub. Review vẫn được lưu, nhưng chưa thể tạo yêu cầu trên GitHub.")

    st.markdown('<div class="review-section-title">Dữ liệu feedback</div>', unsafe_allow_html=True)
    st.write(log.get("feedback") or "Chưa có feedback")

    st.markdown('<div class="review-section-title">Ngữ cảnh truy xuất</div>', unsafe_allow_html=True)
    if contexts:
        st.dataframe(_context_rows(contexts), use_container_width=True, hide_index=True)
    else:
        st.caption("Không có ngữ cảnh")


def render_review_data():
    """Render a stable native Streamlit review workflow."""
    st.title("Dữ liệu review")
    st.caption("Duyệt câu hỏi và câu trả lời đã thu thập. Giao diện này dùng Streamlit thuần để ổn định, dễ đọc và dễ thao tác.")

    filters = st.columns([1, 1, 2, 1])
    with filters[0]:
        status_filter = st.selectbox(
            "Trạng thái",
            ["All", "raw", "approved", "edited", "rejected"],
            format_func=_status_label,
            key="review_status_filter",
        )
    with filters[1]:
        rating_filter = st.selectbox(
            "Đánh giá",
            ["All", "up", "down", "wrong"],
            format_func=_rating_label,
            key="review_rating_filter",
        )
    with filters[2]:
        search_text = st.text_input(
            "Tìm kiếm",
            placeholder="Tìm theo câu hỏi, câu trả lời hoặc tên file nguồn",
            key="review_search_filter",
        )
    with filters[3]:
        limit = st.number_input(
            "Số bản ghi",
            min_value=20,
            max_value=300,
            value=100,
            step=20,
            key="review_limit",
        )

    logs = list_chat_logs(
        limit=int(limit),
        review_status=None if status_filter == "All" else status_filter,
        rating=None if rating_filter == "All" else rating_filter,
    )
    if search_text:
        keyword = search_text.strip().lower()
        logs = [
            log
            for log in logs
            if keyword in (log.get("user_query") or "").lower()
            or keyword in (log.get("assistant_answer") or "").lower()
            or keyword in " ".join(_context_filename(context) for context in log.get("contexts") or []).lower()
        ]

    if not logs:
        st.info("Không tìm thấy dữ liệu chat.")
        return

    status_counts = _count_by(logs, "review_status", default="raw")
    metrics = st.columns(5)
    metrics[0].metric("Tổng", len(logs))
    metrics[1].metric("Chưa review", status_counts.get("raw", 0))
    metrics[2].metric("Đã duyệt", status_counts.get("approved", 0))
    metrics[3].metric("Đã chỉnh sửa", status_counts.get("edited", 0))
    metrics[4].metric("Từ chối", status_counts.get("rejected", 0))

    if "review_current_index" not in st.session_state:
        st.session_state.review_current_index = 0
    if st.session_state.review_current_index >= len(logs):
        st.session_state.review_current_index = max(len(logs) - 1, 0)

    nav = st.columns([1, 1, 2, 1, 1])
    with nav[0]:
        if st.button("Trước", use_container_width=True, disabled=st.session_state.review_current_index <= 0):
            st.session_state.review_current_index -= 1
            st.rerun()
    with nav[1]:
        if st.button("Tiếp", use_container_width=True, disabled=st.session_state.review_current_index >= len(logs) - 1):
            st.session_state.review_current_index += 1
            st.rerun()
    with nav[2]:
        st.progress((st.session_state.review_current_index + 1) / len(logs))
        st.caption(f"Bản ghi {st.session_state.review_current_index + 1} / {len(logs)}")
    with nav[3]:
        if st.button("Làm mới", use_container_width=True):
            st.rerun()
    with nav[4]:
        if st.button("Về đầu", use_container_width=True):
            st.session_state.review_current_index = 0
            st.rerun()

    st.divider()

    selected_index = st.session_state.review_current_index
    log = logs[selected_index]
    turn_id = str(log.get("turn_id") or log.get("document_id") or selected_index)
    current_status = log.get("review_status") or "raw"
    rating = _feedback_rating(log)
    response_type = log.get("response_type") or "unknown"
    contexts = log.get("contexts") or []

    issue_types = ["missing_data", "wrong_data", "outdated_data", "incomplete_answer", "wrong_source", "other"]
    corrected_key = f"corrected_answer_{turn_id}"
    issue_type_key = f"issue_type_{turn_id}"
    status_key = f"review_status_{turn_id}"
    needs_update_key = f"needs_data_update_{turn_id}"
    note_key = f"data_update_note_{turn_id}"

    st.session_state.setdefault(corrected_key, log.get("corrected_answer") or "")
    st.session_state.setdefault(
        issue_type_key,
        log.get("error_type") if log.get("error_type") in issue_types else "missing_data",
    )
    st.session_state.setdefault(status_key, current_status)
    st.session_state.setdefault(needs_update_key, bool(log.get("needs_data_update")))
    st.session_state.setdefault(note_key, log.get("data_update_note") or "")

    def submit_review(status_to_save=None, force_data_update=None):
        final_status = status_to_save or st.session_state.get(status_key, current_status)
        final_needs_data_update = (
            st.session_state.get(needs_update_key, False)
            if force_data_update is None
            else force_data_update
        )
        final_corrected_answer = st.session_state.get(corrected_key, "")
        final_issue_type = st.session_state.get(issue_type_key, "missing_data")
        final_request_note = st.session_state.get(note_key, "")
        request_id = log.get("data_update_request_id")
        github_issue_url = log.get("github_issue_url")
        github_issue_number = log.get("github_issue_number")
        github_project_item_id = log.get("github_project_item_id")
        github_project_error = None
        github_error = None

        if final_needs_data_update and not github_issue_url:
            request_payload = _build_data_update_payload(
                log=log,
                turn_id=turn_id,
                selected_status=final_status,
                corrected_answer=final_corrected_answer,
                issue_type=final_issue_type,
                request_note=final_request_note,
            )
            if is_github_issue_enabled():
                issue_result = create_github_issue(request_payload)
                if issue_result.get("ok"):
                    github_issue_url = issue_result.get("url")
                    github_issue_number = issue_result.get("number")
                    request_payload["github_issue_url"] = github_issue_url
                    request_payload["github_issue_number"] = github_issue_number
                    github_project_item_id = issue_result.get("project_item_id")
                    github_project_error = issue_result.get("project_error")
                    if github_project_item_id:
                        request_payload["github_project_item_id"] = github_project_item_id
                    if github_project_error:
                        request_payload["github_project_error"] = github_project_error
                else:
                    github_error = issue_result.get("error") or "Không thể tạo GitHub Issue."
                    request_payload["github_error"] = github_error
            else:
                github_error = "GitHub chưa được cấu hình."
                request_payload["github_error"] = github_error
            if not request_id:
                request_id = create_data_update_request(request_payload)

        saved = update_review_status(
            turn_id=turn_id,
            review_status=final_status,
            corrected_answer=final_corrected_answer or None,
            error_type=final_issue_type if final_needs_data_update else None,
            needs_data_update=final_needs_data_update,
            data_update_note=final_request_note if final_needs_data_update else None,
            data_update_request_id=request_id,
            github_issue_url=github_issue_url,
            github_issue_number=github_issue_number,
            github_project_item_id=github_project_item_id,
            github_project_error=github_project_error,
        )
        if not saved:
            st.error("Không thể lưu review.")
            return
        if final_needs_data_update and github_issue_url and github_project_error:
            st.warning(
                f"Da luu review va tao GitHub Issue, nhung chua dua duoc vao Project: {github_project_error}"
            )
        elif final_needs_data_update and github_issue_url:
            st.success(f"Đã lưu review và gửi yêu cầu cho đội kỹ thuật: {github_issue_url}")
        elif final_needs_data_update and github_error:
            st.warning(f"Đã lưu review, nhưng chưa gửi được GitHub: {github_error}")
        else:
            st.success("Đã lưu review.")
        if st.session_state.review_current_index < len(logs) - 1:
            st.session_state.review_current_index += 1
        st.rerun()

    meta = st.columns(4)
    meta[0].metric("Trạng thái", _status_label(current_status))
    meta[1].metric("Đánh giá", _rating_label(rating))
    meta[2].metric("Loại phản hồi", _response_type_label(response_type))
    meta[3].metric("Ngữ cảnh", len(contexts))

    st.info(_review_action_hint(log))

    actions = st.columns([1.2, 1, 1.2, 1, 1])
    with actions[0]:
        if st.button("Đúng, lưu & tiếp", use_container_width=True, key=f"approve_{turn_id}"):
            submit_review(status_to_save="approved", force_data_update=False)
    with actions[1]:
        if st.button("Cần sửa", use_container_width=True, key=f"edit_{turn_id}"):
            submit_review(status_to_save="edited")
    with actions[2]:
        if st.button("Gửi đội kỹ thuật", use_container_width=True, key=f"tech_{turn_id}"):
            submit_review(status_to_save="edited", force_data_update=True)
    with actions[3]:
        if st.button("Lưu lựa chọn", use_container_width=True, key=f"save_{turn_id}"):
            submit_review()
    with actions[4]:
        if st.button("Bỏ qua", use_container_width=True, key=f"skip_{turn_id}"):
            if st.session_state.review_current_index < len(logs) - 1:
                st.session_state.review_current_index += 1
                st.rerun()

    st.subheader("Câu hỏi của người dùng")
    st.write(log.get("user_query") or "Chưa có câu hỏi")
    st.subheader("Câu trả lời của chatbot")
    st.write(log.get("assistant_answer") or "Chưa có câu trả lời")

    st.subheader("Thông tin review")
    st.text_area("Câu trả lời đã chỉnh sửa", key=corrected_key, height=160)
    form_cols = st.columns([1, 1, 1.2])
    with form_cols[0]:
        st.selectbox("Lý do cần xử lý", issue_types, format_func=_data_issue_label, key=issue_type_key)
    with form_cols[1]:
        st.selectbox("Trạng thái review", ["raw", "approved", "edited", "rejected"], format_func=_status_label, key=status_key)
    with form_cols[2]:
        st.write("")
        st.write("")
        st.checkbox("Gửi đội kỹ thuật xử lý dữ liệu", key=needs_update_key)

    if st.session_state.get(needs_update_key):
        st.text_area(
            "Ghi chú cho đội kỹ thuật",
            placeholder="Ví dụ: cần bổ sung quy định mới, bot lấy sai nguồn, câu trả lời thiếu bước...",
            key=note_key,
            height=90,
        )
        if log.get("github_issue_url"):
            st.info(f"Yêu cầu đã được gửi: {log.get('github_issue_url')}")
            if log.get("github_project_item_id"):
                st.caption("Yêu cầu đã được đưa vào GitHub Project.")
            elif is_github_project_enabled() and log.get("github_project_error"):
                st.warning(f"Chưa đưa được yêu cầu vào GitHub Project: {log.get('github_project_error')}")
            elif is_github_issue_enabled() and not is_github_project_enabled():
                st.caption("GitHub Project chưa cấu hình; yêu cầu vẫn đã được gửi bằng GitHub Issue.")
        elif not is_github_issue_enabled():
            st.warning("Chưa cấu hình GitHub. Review vẫn được lưu, nhưng chưa thể tạo yêu cầu trên GitHub.")

    detail_tabs = st.tabs(["Nguồn truy xuất", "Feedback"])
    with detail_tabs[0]:
        if contexts:
            st.dataframe(_context_rows(contexts), use_container_width=True, hide_index=True)
        else:
            st.caption("Không có ngữ cảnh.")
    with detail_tabs[1]:
        st.write(log.get("feedback") or "Chưa có feedback")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = set()

page = st.sidebar.selectbox(
    "Chế độ",
    ["Chatbot", "Review data"],
    format_func=lambda value: "Dữ liệu review" if value == "Review data" else value,
    key="app_page",
)
if page == "Review data":
    render_review_data()
    st.stop()

retriever, reasoning, llm_client, chunks = load_rag_components_v2()

# Giao diện chính
st.title("🚓 Trợ lý Ảo - Công an xã An Viễn")
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
        model=settings.OPENAI_MODEL if settings.LLM_PROVIDER == "openai" else settings.GEMINI_MODEL,
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
            model_name = settings.OPENAI_MODEL if settings.LLM_PROVIDER == "openai" else settings.GEMINI_MODEL
            
            # Lọc lấy lịch sử chat (loại bỏ trường 'contexts' và câu hỏi hiện tại)
            history_messages = [
                {"role": msg["role"], "content": msg["content"]} 
                for msg in st.session_state.messages[:-1]
            ]
            # Nối lịch sử với câu hỏi hiện tại (đã kèm ngữ cảnh tài liệu)
            messages_for_llm = history_messages + [{"role": "user", "content": prompt_text}]

            stream = await llm_client.chat.completions.create(
                model=model_name,
                messages=messages_for_llm,
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
