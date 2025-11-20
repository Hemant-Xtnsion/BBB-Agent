import json
import os

try:
    with open('data/products.json', 'r', encoding='utf-8') as f:
        products = json.load(f)
        print(f"Total products: {len(products)}")
        for i, p in enumerate(products[:3]):
            print(f"\nProduct {i+1}:")
            print(f"Title: {p.get('title')}")
            print(f"Thumbnail: {p.get('thumbnail')}")
except Exception as e:
    print(f"Error: {e}")
