from pydantic import BaseModel
from typing import Optional, Dict


class ChatSessionResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str

class ThoughtResponse(BaseModel):
    id: str
    created_at: str


class VisualizeThought(BaseModel):
    id: str
    title: str
    excerpt: str
    created_at: str
    position: list[float]
    label: int

class VisualizeThoughtResponse(BaseModel):
    thoughts: list[VisualizeThought]
    
class ThoughtResponseFull(BaseModel):
    id: str
    title: str
    text_content: str
    image_url: Optional[str]
    audio_url: Optional[str]
    full_content: str
    created_at: str
    updated_at: str