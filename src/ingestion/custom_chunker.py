import re
from configs.settings import settings
from typing import List

def chunk_text(text: str, chunk_size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP) -> List[str]:
    # Split text into paragraphs by newlines
    paragraphs = re.split(r'\n+', text)
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # Force a break if a new "Điều" starts and current chunk is > 50% full
        if re.match(r'^Điều\s+\d+', para) and len(current_chunk) > chunk_size * 0.5:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n"
            continue
            
        # If the paragraph itself is larger than chunk_size, split by sentences
        if len(para) > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                if len(current_chunk) + len(sent) + 1 <= chunk_size:
                    current_chunk += (sent + " ") if current_chunk else sent
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent + " "
        else:
            if len(current_chunk) + len(para) + 1 <= chunk_size:
                current_chunk += (para + "\n") if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n"
                
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks