import os
from typing import List, Dict
from pypdf import PdfReader
import docx2txt
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

def load_pdf(file_path: str) -> str:
    text = ""
    if pdfplumber is not None:
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            pass
    
    if not text.strip():
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            pass
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
                    try:
                        print(f"Error loading {file_path}: {e}")
                    except UnicodeEncodeError:
                        print(f"Error loading {ascii(file_path)}: {ascii(str(e))}")
    return documents
