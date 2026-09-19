"""GitHub issue integration for reviewed data update requests."""

from __future__ import annotations

import os
from typing import Any

import requests


def _shorten(value: str | None, limit: int = 90) -> str:
    text = (value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


def _github_labels() -> list[str]:
    raw_labels = os.getenv("GITHUB_DATA_UPDATE_LABELS", "data-update,rag,needs-review")
    return [label.strip() for label in raw_labels.split(",") if label.strip()]


def is_github_issue_enabled() -> bool:
    return bool(
        os.getenv("GITHUB_TOKEN", "").strip()
        and os.getenv("GITHUB_REPOSITORY", "").strip()
    )


def is_github_project_enabled() -> bool:
    return bool(
        os.getenv("GITHUB_TOKEN", "").strip()
        and os.getenv("GITHUB_PROJECT_ID", "").strip()
    )


def _github_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _graphql_url(api_url: str) -> str:
    return f"{api_url.rstrip('/')}/graphql"


def build_issue_body(payload: dict[str, Any]) -> str:
    contexts = payload.get("contexts") or []
    context_lines = []
    for index, context in enumerate(contexts[:5], start=1):
        metadata = context.get("metadata", {}) if isinstance(context, dict) else {}
        filename = metadata.get("filename") or metadata.get("source_id") or "unknown"
        score = context.get("score") if isinstance(context, dict) else None
        context_lines.append(f"{index}. `{filename}` - score: `{score}`")
    if not context_lines:
        context_lines.append("Không có ngữ cảnh được truy xuất.")

    return "\n".join(
        [
            "## Vấn đề cần xử lý",
            payload.get("request_note") or "Quản trị viên yêu cầu kiểm tra/cập nhật dữ liệu trả lời.",
            "",
            "## Câu hỏi của người dùng",
            payload.get("user_query") or "_Không có_",
            "",
            "## Câu trả lời hiện tại của chatbot",
            payload.get("assistant_answer") or "_Không có_",
            "",
            "## Câu trả lời mong muốn / nội dung đã chỉnh sửa",
            payload.get("corrected_answer") or "_Chưa nhập_",
            "",
            "## Thông tin review",
            f"- Trạng thái review: `{payload.get('review_status') or 'raw'}`",
            f"- Loại lỗi: `{payload.get('error_type') or 'not_set'}`",
            f"- Turn ID: `{payload.get('source_turn_id') or 'unknown'}`",
            f"- Document ID: `{payload.get('source_document_id') or 'unknown'}`",
            "",
            "## Ngữ cảnh chatbot đã dùng",
            *context_lines,
            "",
            "## Việc cần làm",
            "- [ ] Kiểm tra dữ liệu nguồn liên quan",
            "- [ ] Cập nhật hoặc bổ sung dữ liệu nếu cần",
            "- [ ] Tạo phiên bản dữ liệu mới",
            "- [ ] Kiểm tra phiên bản mới trước khi đưa vào sử dụng",
            "- [ ] Cập nhật trạng thái yêu cầu trong Firebase",
        ]
    )


def add_issue_to_project(issue_node_id: str) -> dict[str, Any]:
    """Add an issue to a GitHub Projects v2 board when configured."""
    token = os.getenv("GITHUB_TOKEN", "").strip()
    project_id = os.getenv("GITHUB_PROJECT_ID", "").strip()
    if not token or not project_id:
        return {
            "ok": False,
            "skipped": True,
            "error": "GITHUB_PROJECT_ID chua duoc cau hinh.",
        }
    if not issue_node_id:
        return {
            "ok": False,
            "error": "GitHub issue khong co node_id de dua vao Project.",
        }

    api_url = os.getenv("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    query = """
    mutation($projectId: ID!, $contentId: ID!) {
      addProjectV2ItemById(input: {projectId: $projectId, contentId: $contentId}) {
        item {
          id
        }
      }
    }
    """
    try:
        response = requests.post(
            _graphql_url(api_url),
            headers=_github_headers(token),
            json={
                "query": query,
                "variables": {
                    "projectId": project_id,
                    "contentId": issue_node_id,
                },
            },
            timeout=20,
        )
    except requests.RequestException as exc:
        return {
            "ok": False,
            "error": f"Khong the ket noi GitHub Project: {exc}",
        }

    if response.status_code >= 400:
        return {
            "ok": False,
            "error": f"GitHub Project tra ve loi {response.status_code}: {response.text[:500]}",
        }

    result = response.json()
    if result.get("errors"):
        return {
            "ok": False,
            "error": str(result.get("errors")[:2]),
        }

    item = (
        result.get("data", {})
        .get("addProjectV2ItemById", {})
        .get("item", {})
    )
    return {
        "ok": True,
        "project_item_id": item.get("id"),
    }


def create_github_issue(payload: dict[str, Any]) -> dict[str, Any]:
    """Create a GitHub issue and return issue metadata.

    Required environment variables:
    - GITHUB_TOKEN
    - GITHUB_REPOSITORY, for example "owner/repo"
    """
    token = os.getenv("GITHUB_TOKEN", "").strip()
    repository = os.getenv("GITHUB_REPOSITORY", "").strip()
    if not token or not repository:
        return {
            "ok": False,
            "error": "GITHUB_TOKEN hoặc GITHUB_REPOSITORY chưa được cấu hình.",
        }

    api_url = os.getenv("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    issue_title = "[Yêu cầu dữ liệu] " + _shorten(payload.get("user_query"), limit=100)
    if issue_title.strip() == "[Yêu cầu dữ liệu]":
        issue_title = "[Yêu cầu dữ liệu] Cần kiểm tra câu trả lời chatbot"

    try:
        response = requests.post(
            f"{api_url}/repos/{repository}/issues",
            headers=_github_headers(token),
            json={
                "title": issue_title,
                "body": build_issue_body(payload),
                "labels": _github_labels(),
            },
            timeout=20,
        )
    except requests.RequestException as exc:
        return {
            "ok": False,
            "error": f"Không thể kết nối GitHub: {exc}",
        }
    if response.status_code >= 400:
        return {
            "ok": False,
            "error": f"GitHub trả về lỗi {response.status_code}: {response.text[:500]}",
        }

    issue = response.json()
    result = {
        "ok": True,
        "number": issue.get("number"),
        "url": issue.get("html_url"),
        "api_url": issue.get("url"),
        "node_id": issue.get("node_id"),
    }
    if is_github_project_enabled():
        project_result = add_issue_to_project(issue.get("node_id"))
        if project_result.get("ok"):
            result["project_item_id"] = project_result.get("project_item_id")
        else:
            result["project_error"] = (
                project_result.get("error")
                or "Khong the dua issue vao GitHub Project."
            )
    return result
