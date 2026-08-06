import re
from configs.settings import settings
from typing import List 

def chunk_text(text: str, chunk_size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) < chunk_size:
            current += sent + " "
        else:
            chunks.append(current.strip())
            overlap_text = current.split()[-overlap:] if overlap > 0 else []
            current = " ".join(overlap_text) + " " + sent + " "
    if current:
        chunks.append(current.strip())
    return chunks