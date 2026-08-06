from typing import List, Dict

class Grader:
    @staticmethod
    def is_sufficient(contexts: List[Dict], query: str) -> bool:
        """
        Đánh giá mức độ đủ của ngữ cảnh.
        Heuristic: nếu có ít nhất một context với score > threshold và
        nếu query yêu cầu cụ thể, kiểm tra xem context có chứa từ khóa quan trọng không.
        """
        if not contexts:
            return False
        # Nếu có ít nhất một context có score > 0.5 (tạm đặt)
        if any(c.get('score', 0) > 0.5 for c in contexts):
            return True
        # Nếu query ngắn hoặc mơ hồ, coi là đủ nếu có bất kỳ context
        if len(query.split()) < 5:
            return True
        # Nếu không, có thể yêu cầu thêm
        return False