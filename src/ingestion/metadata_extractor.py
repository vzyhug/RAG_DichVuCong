import re
from typing import Dict

def extract_metadata_from_file(relative_path: str, content: str) -> Dict:
    """
    Trích xuất metadata từ đường dẫn file và nội dung văn bản.
    Ví dụ: SRC-A/01.pdf -> loại: Luật, số hiệu: 68/2020/QH14, ...
    """
    metadata = {}
    # Tách từ đường dẫn
    relative_path = relative_path.replace('\\', '/')
    path_parts = relative_path.split('/')
    filename = path_parts[-1]
    metadata['filename'] = filename
    if len(path_parts) >= 2:
        source_group = path_parts[-2] if "SRC" in path_parts[-2] else path_parts[0]
        metadata['source_group'] = source_group

    # Trích xuất số hiệu văn bản từ tên file hoặc nội dung
    pattern = r'\b(\d+/\d+/(?:QH\d+|NĐ-CP|TT-BCA|CT|NQ|VBHN|...))\b'
    matches = re.findall(pattern, content)
    if matches:
        metadata['document_number'] = matches[0]
    else:
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
    elif 'quyết định' in content.lower() or 'QĐ' in filename:
        metadata['doc_type'] = 'Quyết định'
    elif 'qcvn' in content.lower() or 'qcvn' in filename.lower() or 'tcvn' in filename.lower():
        metadata['doc_type'] = 'Tiêu chuẩn/Quy chuẩn'
    else:
        metadata['doc_type'] = 'Khác'

    # Phân loại chuyên mục (Category)
    content_lower = content.lower()
    filename_lower = filename.lower()
    
    if any(kw in content_lower or kw in filename_lower for kw in ['pccc', 'chữa cháy', 'thoát nạn', 'tcvn 3890', 'tcvn 5738', 'tcvn 7568', 'qcvn 06', 'qcvn 10']):
        metadata['category'] = 'PCCC'
    elif any(kw in content_lower or kw in filename_lower for kw in ['cư trú', 'thường trú', 'tạm trú', 'vneid', 'căn cước']):
        metadata['category'] = 'Cư trú'
    elif any(kw in content_lower or kw in filename_lower for kw in ['đăng ký xe', 'biển số', 'giấy phép lái xe', 'xe máy']):
        metadata['category'] = 'Giao thông'
    elif any(kw in content_lower or kw in filename_lower for kw in ['an ninh', 'trật tự', 'kinh doanh có điều kiện', 'tội phạm', 'lừa đảo', 'trộm cắp']):
        metadata['category'] = 'An ninh trật tự'
    elif any(kw in filename_lower for kw in ['anvien', 'an viên', 'liên hệ', 'lịch làm việc']):
        metadata['category'] = 'Thông tin địa phương'
    else:
        metadata['category'] = 'Chung'

    return metadata