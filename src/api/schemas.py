from pydantic import BaseModel
from typing import Optional, List

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    context_used: Optional[List[dict]] = None
    need_clarification: bool = False
    clarification_question: Optional[str] = None
    emergency: bool = False