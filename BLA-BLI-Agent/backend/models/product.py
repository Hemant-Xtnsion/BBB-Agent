from pydantic import BaseModel, HttpUrl
from typing import Optional, List


class Product(BaseModel):
    """Product model representing a product from blabliblulife.com"""
    title: str
    price: str
    thumbnail: str
    url: str
    description: Optional[str] = ""
    tags: Optional[List[str]] = []
    category: Optional[str] = ""
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Organic Cotton T-Shirt",
                "price": "$29.99",
                "thumbnail": "https://example.com/image.jpg",
                "url": "https://blabliblulife.com/product/123",
                "description": "Comfortable organic cotton t-shirt",
                "tags": ["clothing", "organic", "sustainable"],
                "category": "Apparel"
            }
        }


class ButtonSuggestion(BaseModel):
    """Button suggestion for user interaction"""
    label: str
    action: str
    type: str  # "primary", "secondary", "success", "danger"


class ChatMessage(BaseModel):
    """Chat message from user"""
    message: str
    conversation_id: Optional[str] = None
    email: Optional[str] = None
    order_number: Optional[str] = None


class ChatResponse(BaseModel):
    """Response from chat endpoint"""
    type: str  # "text" or "products"
    message: str
    products: Optional[List[Product]] = []
    buttons: Optional[List[ButtonSuggestion]] = []
    state: Optional[dict] = None  # For order tracking state
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "products",
                "message": "Here are some sustainable clothing options:",
                "products": [],
                "buttons": []
            }
        }
