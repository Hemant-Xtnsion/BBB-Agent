import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from recommender import PERFUME_CATALOG, extract_preferences_and_recommend
    
    print(f"Catalog loaded: {len(PERFUME_CATALOG)} products")
    
    if PERFUME_CATALOG:
        print(f"\nFirst product: {PERFUME_CATALOG[0].get('name')}")
        print(f"Gender: {PERFUME_CATALOG[0].get('gender_profile')}")
        print(f"Price: {PERFUME_CATALOG[0].get('price')}")
        
        # Test a simple query
        print("\n" + "="*50)
        print("Testing: 'I need a perfume for men'")
        print("="*50)
        
        prefs, recs = extract_preferences_and_recommend("I need a perfume for men", limit=2)
        
        print(f"\nPreferences detected: {prefs}")
        print(f"\nRecommendations: {len(recs)}")
        for i, rec in enumerate(recs, 1):
            print(f"  {i}. {rec.get('name')} - Score: {rec.get('score')}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
