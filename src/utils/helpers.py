import re
from unidecode import unidecode

def normalize_vietnamese(text: str) -> str:
    text = unidecode(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()