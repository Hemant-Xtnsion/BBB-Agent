from typing import List, Optional, Any, Dict
from pydantic import BaseModel


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    email: Optional[str] = None
    order_number: Optional[str] = None
    conversation_id: Optional[str] = None  # Keep for compatibility with existing frontend


class ButtonSuggestion(BaseModel):
    label: str
    action: str
    type: str  # "primary", "secondary", "success", "danger"


class ChatResponse(BaseModel):
    reply: str
    state: dict
    products: Optional[List[Dict[str, Any]]] = None
    buttons: Optional[List[ButtonSuggestion]] = None
    # Keep existing fields for backward compatibility
    type: Optional[str] = None  # "text" or "products"
    message: Optional[str] = None  # Alias for reply

