import json
from pathlib import Path
from typing import List, Dict


def load_products(file_path: str = "data/products.json") -> List[Dict]:
    """Load products from JSON file"""
    path = Path(file_path)
    
    if not path.exists():
        print(f"[!] Products file not found at {file_path}")
        return []
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            products = json.load(f)
        print(f"[+] Loaded {len(products)} products from {file_path}")
        return products
    except Exception as e:
        print(f"[x] Error loading products: {e}")
        return []


def save_products(products: List[Dict], file_path: str = "backend/data/products.json"):
    """Save products to JSON file"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(products, f, indent=2, ensure_ascii=False)
        print(f"[+] Saved {len(products)} products to {file_path}")
    except Exception as e:
        print(f"[x] Error saving products: {e}")
