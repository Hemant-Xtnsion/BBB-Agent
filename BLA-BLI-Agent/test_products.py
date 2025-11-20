import json
from pathlib import Path

# Load products
data_path = Path(__file__).parent / "backend" / "data" / "products.json"
data = json.loads(data_path.read_text(encoding='utf-8'))

print(f"✅ Successfully loaded {len(data)} products\n")

# Count by gender_profile
for_him = sum(1 for p in data if p.get('gender_profile') == 'for_him')
for_her = sum(1 for p in data if p.get('gender_profile') == 'for_her')
unisex = sum(1 for p in data if p.get('gender_profile') == 'unisex')

print("Gender Breakdown:")
print(f"  For Him: {for_him}")
print(f"  For Her: {for_her}")
print(f"  Unisex: {unisex}")

# Count by category
categories = {}
for p in data:
    cat = p.get('category', 'Unknown')
    categories[cat] = categories.get(cat, 0) + 1

print("\nCategory Breakdown:")
for cat, count in sorted(categories.items()):
    print(f"  {cat}: {count}")

# Count best sellers
best_sellers = sum(1 for p in data if 'best_seller' in p.get('collection_tags', []))
print(f"\nBest Sellers: {best_sellers}")

# Show sample products
print("\n📦 Sample Products:")
for p in data[:3]:
    print(f"\n  • {p.get('name')}")
    print(f"    Gender: {p.get('gender_profile')}")
    print(f"    Price: ₹{p.get('price')}")
    print(f"    Vibes: {', '.join(p.get('vibe_tags', [])[:3])}")
    print(f"    Occasions: {', '.join(p.get('occasion_tags', [])[:2])}")

print("\n✅ All data loaded successfully!")
