import json
from pathlib import Path

products_path = Path(__file__).parent / "backend" / "data" / "products.json"
with open(products_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

if data:
    first = data[0]
    print("All fields in first product:")
    for key in sorted(first.keys()):
        value = first[key]
        if isinstance(value, (list, dict)):
            print(f"  {key}: {type(value).__name__} with {len(value) if isinstance(value, (list, dict)) else 0} items")
        else:
            value_str = str(value)[:50]
            print(f"  {key}: {value_str}")
