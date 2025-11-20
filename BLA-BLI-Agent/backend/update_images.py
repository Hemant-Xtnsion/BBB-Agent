import json
import os

file_path = 'data/products.json'
image_path = 'http://localhost:5173/perfume_placeholder.png'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        products = json.load(f)
    
    print(f"Updating {len(products)} products...")
    
    for product in products:
        product['thumbnail'] = image_path
        
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
        
    print("[+] Updated all products to use placeholder image")
    
except Exception as e:
    print(f"[x] Error: {e}")
