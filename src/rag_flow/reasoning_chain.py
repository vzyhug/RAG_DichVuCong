from typing import Dict, List

class ReasoningChain:
    def __init__(self):
        pass

    def process(self, query: str, intent: str, entities: Dict, chunks: List[Dict]) -> Dict:
        """
        Xác định còn thiếu thông tin không, và đưa ra câu hỏi làm rõ.
        Ở đây ta giả định mỗi chunk có trường 'required_entities' và 'clarifying_question'.
        """
        # Tìm chunk phù hợp nhất dựa trên intent hoặc từ khóa
        selected_chunk = None
        for chunk in chunks:
            if chunk.get('intent_code') == intent:
                selected_chunk = chunk
                break
        if selected_chunk is None:
            return {"ready": False, "clarification": "Anh/chị vui lòng cung cấp thêm thông tin cụ thể hơn."}

        required = selected_chunk.get('required_entities', [])
        if not required:
            return {"ready": True}

        missing = [r for r in required if r not in entities]
        if missing:
            # Lấy câu hỏi làm rõ từ chunk
            clarification = selected_chunk.get('clarifying_question', f"Anh/chị vui lòng cung cấp {', '.join(missing)}.")
            return {"ready": False, "clarification": clarification}
        return {"ready": True}