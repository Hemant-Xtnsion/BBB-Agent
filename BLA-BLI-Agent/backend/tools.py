import json
from pathlib import Path
from typing import List, Dict, Optional

# Load perfume products - try products.json first, fallback to sample
PRODUCTS_PATH = Path(__file__).parent / "data" / "products.json"
SAMPLE_PRODUCTS_PATH = Path(__file__).parent / "data" / "sample_products.json"

try:
    if PRODUCTS_PATH.exists():
        PRODUCTS_DATA = json.loads(PRODUCTS_PATH.read_text(encoding="utf-8"))
    elif SAMPLE_PRODUCTS_PATH.exists():
        PRODUCTS_DATA = json.loads(SAMPLE_PRODUCTS_PATH.read_text(encoding="utf-8"))
    else:
        PRODUCTS_DATA = []
except Exception:
    PRODUCTS_DATA = []

# Load order data (will be created if doesn't exist)
ORDERS_PATH = Path(__file__).parent / "data" / "orders.json"

# Initialize orders data structure if it doesn't exist
def _init_orders_data():
    """Initialize orders.json with sample data if it doesn't exist"""
    if not ORDERS_PATH.exists():
        sample_orders = {
            "customers": [
                {
                    "email": "customer@example.com",
                    "orders": [
                        {
                            "order_number": "BLB-1001",
                            "status": "shipped",
                            "items": [
                                {"name": "Love Drunk - 100ml", "quantity": 1}
                            ],
                            "shipping_address": "123 Main St, City, State 12345",
                            "tracking": {
                                "carrier": "FedEx",
                                "tracking_id": "FX123456789",
                                "eta": "2024-01-15"
                            }
                        }
                    ]
                }
            ],
            "brand": {
                "name": "BlaBli Blu",
                "support_email": "support@blabliblulife.com",
                "return_policy_days": 30,
                "return_policy": "We offer a 30-day return policy. If you're not satisfied, return within 30 days for a full refund.",
                "shipping_policy": "Free shipping on orders over ₹500. Dispatch in 1-2 business days."
            }
        }
        ORDERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        ORDERS_PATH.write_text(json.dumps(sample_orders, indent=2), encoding="utf-8")
    return json.loads(ORDERS_PATH.read_text(encoding="utf-8"))

ORDERS_DATA = _init_orders_data()

# for /public-data route
INIT_PUBLIC_DATA = {
    "brand": ORDERS_DATA.get("brand", {}),
    "products": [
        {
            "id": p.get("id", idx),
            "name": p.get("name", ""),
            "price": p.get("price", 0),
            "short_description": p.get("short_description", ""),
            "image_url": p.get("image_url", ""),
            "product_url": p.get("product_url", ""),
        }
        for idx, p in enumerate(PRODUCTS_DATA)
    ],
}


def list_products():
    """List all products"""
    return PRODUCTS_DATA


def get_product_by_name(name: str):
    """Get a product by name (exact or partial match)"""
    name = name.lower().strip()
    # Try exact match first
    for p in PRODUCTS_DATA:
        if p.get("name", "").lower() == name:
            return p
    # Try partial match
    for p in PRODUCTS_DATA:
        if name in p.get("name", "").lower():
            return p
    # Try handle match
    for p in PRODUCTS_DATA:
        if name in p.get("handle", "").lower():
            return p
    return None


def recommend_products(description: str):
    """Enhanced product recommendation based on keywords and tags"""
    desc = description.lower()
    scored = []
    
    for p in PRODUCTS_DATA:
        score = 0
        name = p.get("name", "").lower()
        gender_profile = p.get("gender_profile", "")
        vibe_tags = p.get("vibe_tags", [])
        occasion_tags = p.get("occasion_tags", [])
        collection_tags = p.get("collection_tags", [])
        notes = p.get("notes", {})
        
        # Gender matching (high priority)
        if any(word in desc for word in ["men", "male", "him", "guy"]):
            if gender_profile == "for_him":
                score += 5
        if any(word in desc for word in ["women", "female", "her", "lady"]):
            if gender_profile == "for_her":
                score += 5
        if "unisex" in desc:
            if gender_profile == "unisex":
                score += 5
        
        # Scent type matching from vibe_tags
        scent_keywords = {
            "fresh": ["fresh", "citrus", "beach", "clean", "breezy"],
            "floral": ["floral", "rose", "jasmine", "flower"],
            "woody": ["woody", "wood", "cedar", "vetiver"],
            "oud": ["oud", "habibi"],
            "sweet": ["sweet", "honey", "vanilla", "cookie", "gourmand"],
            "spicy": ["spicy", "cinnamon", "cardamom", "warm"],
        }
        
        for scent_type, keywords in scent_keywords.items():
            if any(kw in desc for kw in keywords):
                # Check vibe_tags
                if scent_type in [tag.lower() for tag in vibe_tags]:
                    score += 4
                # Check notes
                all_notes = notes.get("head", []) + notes.get("heart", []) + notes.get("base", [])
                all_notes_lower = [note.lower() for note in all_notes]
                if any(kw in all_notes_lower for kw in keywords):
                    score += 3
        
        # Occasion matching
        occasion_keywords = {
            "date": ["date_night", "romantic", "intimate"],
            "work": ["work", "office", "formal", "business", "boardroom"],
            "party": ["party", "clubbing", "night_out"],
            "casual": ["casual", "daily_wear", "daytime"],
        }
        
        for occ_word, occ_tags in occasion_keywords.items():
            if occ_word in desc:
                if any(tag in occasion_tags for tag in occ_tags):
                    score += 3
        
        # Budget matching
        price = p.get("price", 0)
        if any(word in desc for word in ["budget", "cheap", "affordable", "trial"]):
            if price <= 500:
                score += 3
        elif any(word in desc for word in ["premium", "luxury", "high end", "best"]):
            if price >= 750:
                score += 3
        
        # Bonus for best sellers
        if "best_seller" in collection_tags:
            score += 2
        
        scored.append((score, p))
    
    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Get top 2 products with positive scores
    top = [p for s, p in scored if s > 0][:2]
    
    # If no matches, return top 2 best sellers or first 2 products
    if not top:
        best_sellers = [p for p in PRODUCTS_DATA if "best_seller" in p.get("collection_tags", [])]
        top = best_sellers[:2] if best_sellers else PRODUCTS_DATA[:2]
    
    # Ensure all products have required fields
    for product in top:
        if "image_url" not in product:
            product["image_url"] = ""
        if "product_url" not in product:
            product["product_url"] = ""
        if "id" not in product:
            product["id"] = PRODUCTS_DATA.index(product) if product in PRODUCTS_DATA else 0
    
    return top


def get_customer_by_email(email: str):
    """Get customer by email"""
    for c in ORDERS_DATA.get("customers", []):
        if c.get("email", "").lower() == email.lower():
            return c
    return None


def get_order_status(email: str, order_number: str):
    """Get order status by email and order number"""
    cust = get_customer_by_email(email)
    if not cust:
        return {"error": "customer_not_found"}
    
    for o in cust.get("orders", []):
        if o.get("order_number", "").upper() == order_number.upper():
            return {
                "order_number": o.get("order_number"),
                "status": o.get("status"),
                "eta": o.get("tracking", {}).get("eta"),
                "carrier": o.get("tracking", {}).get("carrier"),
                "tracking_id": o.get("tracking", {}).get("tracking_id"),
                "shipping_address": o.get("shipping_address"),
                "items": o.get("items", []),
            }
    return {"error": "order_not_found"}


def get_return_policy():
    """Get return policy"""
    brand = ORDERS_DATA.get("brand", {})
    return {
        "policy": brand.get("return_policy", ""),
        "days": brand.get("return_policy_days", 30),
    }

