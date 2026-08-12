import json
import re

def clean_text(text):
    # 1. Remove Boilerplate
    # Remove phrases like "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Độc lập - Tự do - Hạnh phúc" only if they appear together or are on their own line
    text = re.sub(r'(?i)CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\s*(?:\n|\s)*Độc lập\s*-\s*Tự do\s*-\s*Hạnh phúc', '', text)
    text = re.sub(r'(?im)^\s*CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\s*$', '', text)
    text = re.sub(r'(?im)^\s*Độc lập\s*-\s*Tự do\s*-\s*Hạnh phúc\s*$', '', text)
    
    # Remove "CÔNG BÁO..." strings and adjacent page numbers
    text = re.sub(r'(?i)\b\d*\s*CÔNG BÁO/Số \d+.*?(?:\n|$)', '', text)
    text = re.sub(r'(?i)\b\d*\s*CÔNG BÁO.*?Ngày.*?\d{2}-\d{2}-\d{4}\s*\d*\b', '', text)
    
    # Remove document numbers like "Luật số: 26/2023/QH15"
    text = re.sub(r'(?i)(?:Luật|Nghị định|Quyết định|Thông tư|Chỉ thị)\s+số:\s*\d+/\d+/[A-Z0-9\-]+', '', text)
    
    # Remove solitary QUỐC HỘI
    text = re.sub(r'(?m)^QUỐC HỘI$', '', text)
    
    # 2. Fix line breaks
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if cleaned_lines and not re.search(r'[.,:?!;]\s*$', cleaned_lines[-1]) and not re.search(r'^(?:Điều|Chương|Mục|Phần)\s+\d+', line, re.IGNORECASE):
            # Also check if it's not starting a new structural element like Điều, Chương
            cleaned_lines[-1] = cleaned_lines[-1] + " " + line
        else:
            cleaned_lines.append(line)
            
    # 3. Join and remove redundant spaces
    text = '\n'.join(cleaned_lines)
    
    # Chuẩn hóa khoảng trắng xung quanh dấu =
    text = re.sub(r'\s*=\s*', ' = ', text)
    
    # Remove multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)
    # Tối đa 2 lần xuống dòng
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            data = json.loads(line)
            cleaned_text = clean_text(data['text'])
            data['text'] = cleaned_text
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    input_path = r"E:\nam4_hk1\Intern\DVC-BCA-RAG\data\processed\chunks.jsonl"
    output_path = r"E:\nam4_hk1\Intern\DVC-BCA-RAG\data\processed\cleaned_chunks.jsonl"
    process_file(input_path, output_path)
    print("Processing complete.")
