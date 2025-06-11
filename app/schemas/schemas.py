from pydantic import BaseModel
from typing import Optional, Dict


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