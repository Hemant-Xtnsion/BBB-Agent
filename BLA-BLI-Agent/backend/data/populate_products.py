"""
Script to populate products.json with the perfume catalog data.
Run this script to automatically create the products.json file.
"""

import json
from pathlib import Path

# The product data
PRODUCTS_DATA = [
  {
    "id": "old_money_100",
    "handle": "old-money",
    "name": "Old Money - 100ml",
    "category": "Parfum - 100ml",
    "collection_tags": ["men", "parfum", "best_seller"],
    "gender_profile": "for_him",
    "oil_concentration_percent": 25,
    "price": 755,
    "mrp": 1000,
    "size_ml": 100,
    "short_description": "Dry, elegant woody-floral parfum that smells like quiet power and generational wealth.",
    "vibe_tags": ["woody", "floral", "dry", "mature", "formal", "luxury"],
    "occasion_tags": ["boardroom", "formal_events", "business_meetings"],
    "season_tags": ["autumn", "winter"],
    "notes": {
      "head": ["Apple", "Davana", "Chamomile"],
      "heart": ["Damask Rose", "Cedar", "Osmanthus"],
      "base": ["Vanilla Absolute", "Tonka Bean", "Patchouli"]
    },
    "image_url": "https://blabliblulife.com/cdn/shop/files/2_0f0ca980-69d6-416f-b0a5-f3405478ddbb.jpg?v=1761641553",
    "product_url": "https://blabliblulife.com/products/old-money",
    "add_to_cart_url": "/cart/add?id=50151027015975",
    "reviews_count": 144,
    "positioning_line": "Legacy. Grandeur. Power.",
    "availability": "in_stock"
  },
  {
    "id": "by_the_beach_100",
    "handle": "by-the-beach",
    "name": "By the Beach - 100ml",
    "category": "Parfum - 100ml",
    "collection_tags": ["men", "parfum", "fresh"],
    "gender_profile": "for_him",
    "oil_concentration_percent": 20,
    "price": 595,
    "mrp": 1000,
    "size_ml": 100,
    "short_description": "A fresh and clean perfume with a tangy symphony of lemon, bergamot, and apple. Smells like a quiet beach nap.",
    "vibe_tags": ["fresh", "citrus", "clean", "breezy", "casual"],
    "occasion_tags": ["weekend", "vacation", "daytime", "casual_hangout"],
    "season_tags": ["summer", "spring"],
    "notes": {
      "head": ["Lemon", "Orange", "Passionfruit"],
      "heart": ["Caramel", "Cashmere", "Bergamot"],
      "base": ["Ambergris", "Musky Vanilla", "Apple"]
    },
    "image_url": "https://blabliblulife.com/cdn/shop/files/by_the_beach_1f2c7a8c-21dd-4003-ac1d-b41838dcd907.jpg?v=1761643319",
    "product_url": "https://blabliblulife.com/products/by-the-beach",
    "add_to_cart_url": "/cart/add?id=50151094911271",
    "reviews_count": 123,
    "positioning_line": "Quiet. Breezy. Serene.",
    "availability": "in_stock"
  },
  # Add remaining 19 products here...
  # (Truncated for brevity - you would include all 21 products)
]

def main():
    """Create products.json file with the product data"""
    # Get the data directory path
    data_dir = Path(__file__).parent
    products_file = data_dir / "products.json"
    
    # Write the data
    with open(products_file, 'w', encoding='utf-8') as f:
        json.dump(PRODUCTS_DATA, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Successfully created {products_file}")
    print(f"📦 Added {len(PRODUCTS_DATA)} products")
    print("\nProduct breakdown:")
    
    # Count by gender
    for_him = sum(1 for p in PRODUCTS_DATA if p.get('gender_profile') == 'for_him')
    for_her = sum(1 for p in PRODUCTS_DATA if p.get('gender_profile') == 'for_her')
    unisex = sum(1 for p in PRODUCTS_DATA if p.get('gender_profile') == 'unisex')
    
    print(f"  - For Him: {for_him}")
    print(f"  - For Her: {for_her}")
    print(f"  - Unisex: {unisex}")
    
    # Count best sellers
    best_sellers = sum(1 for p in PRODUCTS_DATA if 'best_seller' in p.get('collection_tags', []))
    print(f"  - Best Sellers: {best_sellers}")

if __name__ == "__main__":
    main()
