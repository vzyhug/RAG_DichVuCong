import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import requests
# pyrefly: ignore [missing-import]
from bs4 import BeautifulSoup
import os
import re
from urllib.parse import urljoin

# 1. Danh sách các danh mục và URL thực tế trên trang Cục Cảnh sát PCCC
danh_muc_urls = {
    "LUAT": "https://canhsatpccc.gov.vn/vi/legal-document/luat",
    "NGHI_DINH": "https://canhsatpccc.gov.vn/vi/legal-document/nghi-dinh",
    "THONG_TU": "https://canhsatpccc.gov.vn/vi/legal-document/thong-tu",
    "TIEU_CHUAN_QUY_CHUAN": "https://canhsatpccc.gov.vn/vi/legal-document/tieu-chuan-quy-chuan",
    "VAN_BAN_DU_THAO": "https://canhsatpccc.gov.vn/vi/legal-document/van-ban-du-thao"
}

# Các định dạng tài liệu cần tải
target_extensions = ('.pdf', '.doc', '.docx', '.xls', '.xlsx')

TARGET_DIR = os.path.join("data/raw/CRAWL-KNOWLEDGE-BASE", "SRC-F")

def get_next_number_and_existing_files(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    existing_files = os.listdir(target_dir)
    max_num = 0
    for f in existing_files:
        match = re.match(r'^(\d+)', f)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return max_num + 1, existing_files

def download_documents_from_category(category_name, url, current_num, existing_files):
    # Thêm User-Agent giả lập trình duyệt
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    }

    try:
        print(f"\n[{category_name}] Đang quét dữ liệu tại: {url}")
        # Tắt cảnh báo verify SSL nếu trang web thỉnh thoảng bị lỗi chứng chỉ (tuỳ chọn verify=False)
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a')
        
        download_count = 0
        for link in links:
            href = link.get('href')
            
            # Kiểm tra xem link có chứa các đuôi file tài liệu không
            if href and any(href.lower().endswith(ext) for ext in target_extensions):
                normalized_href = href.replace('\\', '/')
                file_url = urljoin(url, normalized_href)
                
                # Lấy tên file gốc từ URL
                original_file_name = file_url.split('/')[-1].split('?')[0]
                
                # Bỏ qua nếu file đã được tải xuống từ trước
                already_downloaded = any(original_file_name in f for f in existing_files)
                if already_downloaded:
                    print(f"  -> File đã tồn tại, bỏ qua: {original_file_name}")
                    continue
                
                # Tạo tên file mới có đánh số (ví dụ: 01_LUAT_Luat-PCCC.pdf)
                new_file_name = f"{current_num:02d}_{category_name}_{original_file_name}"
                save_path = os.path.join(TARGET_DIR, new_file_name)
                
                print(f"  -> Đang tải: {new_file_name}...")
                file_response = requests.get(file_url, headers=headers, timeout=15, verify=False)
                
                # Lưu file
                with open(save_path, 'wb') as f:
                    f.write(file_response.content)
                
                existing_files.append(new_file_name)
                download_count += 1
                current_num += 1
                
        print(f"✅ Đã tải xong {download_count} tài liệu cho danh mục {category_name}.")
        return current_num

    except Exception as e:
        print(f"❌ Lỗi khi xử lý danh mục {category_name}: {e}")
        return current_num

if __name__ == "__main__":
    import urllib3
    # Ẩn cảnh báo bảo mật nếu dùng verify=False
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    print("=" * 60)
    print("CÔNG CỤ TỰ ĐỘNG TẢI TÀI LIỆU PCCC & CNCH")
    print("=" * 60)
    
    current_num, existing_files = get_next_number_and_existing_files(TARGET_DIR)

    # Chạy vòng lặp qua từng danh mục để tải
    for ten_danh_muc, duong_dan in danh_muc_urls.items():
        current_num = download_documents_from_category(ten_danh_muc, duong_dan, current_num, existing_files)
        
    print("\n🎉 HOÀN TẤT QUÁ TRÌNH TẢI TÀI LIỆU!")