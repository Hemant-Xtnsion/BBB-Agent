import json
from pathlib import Path

# Load the products.json directly
products_path = Path(__file__).parent / "backend" / "data" / "products.json"
with open(products_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total products: {len(data)}")
print(f"Type: {type(data)}")

if data:
    first = data[0]
    print(f"\nFirst product keys: {list(first.keys())[:10]}")
    print(f"\nFirst product sample:")
    print(f"  id: {first.get('id')}")
    print(f"  name: {first.get('name')}")
    print(f"  gender_profile: {first.get('gender_profile')}")
    print(f"  price: {first.get('price')}")
    print(f"  vibe_tags: {first.get('vibe_tags')}")
    print(f"  occasion_tags: {first.get('occasion_tags')}")
    
    # Check if all products have the required fields
    print(f"\nField coverage:")
    for field in ['name', 'gender_profile', 'price', 'vibe_tags', 'occasion_tags', 'notes']:
        count = sum(1 for p in data if p.get(field))
        print(f"  {field}: {count}/{len(data)}")
