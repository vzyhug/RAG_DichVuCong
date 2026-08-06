import re
from typing import Dict, List, Optional
from src.utils.helpers import normalize_vietnamese

class QueryAnalyzer:
    def __init__(self):
        # Có thể load danh sách từ khóa/tags từ chunk để nhận diện
        pass

    def normalize(self, query: str) -> str:
        return normalize_vietnamese(query)

    def extract_entities(self, query: str) -> Dict[str, str]:
        entities = {}
        # Mã hồ sơ (dạng chữ + số, ít nhất 6 ký tự)
        hs_pattern = r'\b([A-Z0-9]{6,})\b'
        matches = re.findall(hs_pattern, query)
        if matches:
            entities['ma_ho_so'] = matches[0]

        # Số điện thoại
        phone_pattern = r'\b(0[3-9]\d{8,9})\b'
        matches = re.findall(phone_pattern, query)
        if matches:
            entities['so_dien_thoai'] = matches[0]

        # Địa chỉ (có thể dùng entity recognition nâng cao, nhưng tạm thời dùng từ khóa)
        # Ở đây chỉ lấy mẫu đơn giản
        address_keywords = ['phường', 'xã', 'huyện', 'quận', 'thành phố', 'số nhà']
        for kw in address_keywords:
            if kw in query.lower():
                # lấy đoạn quanh từ khóa
                idx = query.lower().find(kw)
                start = max(0, idx - 20)
                end = min(len(query), idx + 30)
                entities['dia_chi'] = query[start:end].strip()
                break

        return entities

    def classify_intent(self, query: str, chunks: List[Dict]) -> Optional[str]:
        # So khớp với câu hỏi mẫu trong chunk (nếu có)
        # Ở đây ta giả định chunks đã được load và có trường 'question_variants'
        # Nếu không có, có thể dùng embedding hoặc từ khóa
        # Thực tế: ta sẽ dùng retriever để lấy top-k, sau đó xác định intent từ chunk có điểm cao nhất
        # Vì vậy hàm này có thể không cần thiết, nhưng để đảm bảo tính module, ta để đây.
        return None