import re
from typing import Dict

def extract_metadata_from_file(relative_path: str, content: str) -> Dict:
    """
    Trích xuất metadata từ đường dẫn file và nội dung văn bản.
    Ví dụ: SRC-A/01.pdf -> loại: Luật, số hiệu: 68/2020/QH14, ...
    """
    metadata = {}
    # Tách từ đường dẫn
    path_parts = relative_path.split('/')
    if len(path_parts) >= 2:
        source_group = path_parts[0]  # SRC-A, SRC-B, ...
        metadata['source_group'] = source_group
        filename = path_parts[-1]
        metadata['filename'] = filename

    # Trích xuất số hiệu văn bản từ tên file hoặc nội dung
    # Mẫu: 68/2020/QH14, 154/2024/NĐ-CP, 66/2023/TT-BCA, ...
    pattern = r'\b(\d+/\d+/(?:QH\d+|NĐ-CP|TT-BCA|CT|NQ|VBHN|...))\b'
    matches = re.findall(pattern, content)
    if matches:
        metadata['document_number'] = matches[0]  # lấy số đầu tiên
    else:
        # thử tìm trong tên file
        file_match = re.search(pattern, filename)
        if file_match:
            metadata['document_number'] = file_match.group(1)

    # Xác định loại văn bản
    if 'luật' in content.lower() or 'QH' in content:
        metadata['doc_type'] = 'Luật'
    elif 'nghị định' in content.lower() or 'NĐ-CP' in content:
        metadata['doc_type'] = 'Nghị định'
    elif 'thông tư' in content.lower() or 'TT-BCA' in content:
        metadata['doc_type'] = 'Thông tư'
    elif 'quyết định' in content.lower():
        metadata['doc_type'] = 'Quyết định'
    else:
        metadata['doc_type'] = 'Khác'

    return metadata