import os
from typing import List, Dict
from pypdf import PdfReader
import docx2txt
from pathlib import Path

def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def load_docx(file_path: str) -> str:
    return docx2txt.process(file_path)

def load_text(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_document(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext == '.pdf':
        return load_pdf(file_path)
    elif ext == '.docx':
        return load_docx(file_path)
    elif ext == '.txt':
        return load_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def load_all_documents(root_dir: str) -> List[Dict]:
    documents = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.pdf', '.docx', '.txt')):
                file_path = os.path.join(dirpath, filename)
                try:
                    content = load_document(file_path)
                    if content.strip():
                        rel_path = os.path.relpath(file_path, root_dir)
                        source_id = rel_path.replace(os.sep, '_').replace('.', '_')
                        documents.append({
                            "source_id": source_id,
                            "file_path": file_path,
                            "relative_path": rel_path,
                            "content": content,
                        })
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return documents