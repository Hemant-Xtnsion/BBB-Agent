from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Tuple, Union

# Try to load products from products.json, fallback to sample if not available
PRODUCTS_PATH = Path(__file__).parent / "data" / "products.json"
SAMPLE_PRODUCTS_PATH = Path(__file__).parent / "data" / "sample_products.json"

try:
    if PRODUCTS_PATH.exists():
        catalog = json.loads(PRODUCTS_PATH.read_text(encoding="utf-8"))
    elif SAMPLE_PRODUCTS_PATH.exists():
        catalog = json.loads(SAMPLE_PRODUCTS_PATH.read_text(encoding="utf-8"))
    else:
        catalog = []
    products = catalog if isinstance(catalog, list) else catalog.get("products", [])
except Exception:
    products = []

# ---------------------------------------------------------------------------
# 1. Perfume Catalog - Direct use of new data structure
# ---------------------------------------------------------------------------

PERFUME_CATALOG: List[Dict[str, Any]] = products


# ---------------------------------------------------------------------------
# 1.5. Build dynamic note vocabulary from catalog
# ---------------------------------------------------------------------------

def _build_note_vocabulary() -> Dict[str, List[str]]:
    """
    Build a dynamic vocabulary of notes from the product catalog.
    Returns a dict mapping note keywords to their normalized forms.
    """
    note_vocab = {}
    
    for product in PERFUME_CATALOG:
        notes = product.get("notes", {})
        all_notes = (
            notes.get("head", []) + 
            notes.get("heart", []) + 
            notes.get("base", [])
        )
        
        for note in all_notes:
            if not note:
                continue
            note_lower = note.lower().strip()
            # Store both full note and individual words
            if note_lower not in note_vocab:
                note_vocab[note_lower] = []
            note_vocab[note_lower].append(note)
            
            # Also index individual words (e.g., "vanilla absolute" -> ["vanilla", "absolute"])
            words = note_lower.split()
            for word in words:
                if len(word) > 2:  # Ignore very short words
                    if word not in note_vocab:
                        note_vocab[word] = []
                    if note not in note_vocab[word]:
                        note_vocab[word].append(note)
    
    return note_vocab


# Cache the note vocabulary
_NOTE_VOCABULARY: Optional[Dict[str, List[str]]] = None

def _get_note_vocabulary() -> Dict[str, List[str]]:
    """Get cached note vocabulary, building it if needed"""
    global _NOTE_VOCABULARY
    if _NOTE_VOCABULARY is None:
        _NOTE_VOCABULARY = _build_note_vocabulary()
    return _NOTE_VOCABULARY


# ---------------------------------------------------------------------------
# 2. Enhanced Preference extraction from user message
# ---------------------------------------------------------------------------

def _detect_gender(text: str) -> Optional[str]:
    """Detect gender preference from user text"""
    text_lower = text.lower()
    # Check for explicit gender mentions
    if any(word in text_lower for word in ["men", "male", "him", "guy", "man's", "masculine", "for him", "boyfriend", "husband"]):
        return "for_him"
    if any(word in text_lower for word in ["women", "female", "her", "lady", "woman's", "feminine", "for her", "girlfriend", "wife"]):
        return "for_her"
    if "unisex" in text_lower or "both" in text_lower:
        return "unisex"
    return None


def _detect_scent_type(text: str) -> List[str]:
    """Detect scent type preferences from user text - returns list of matching types"""
    text_lower = text.lower()
    scent_keywords = {
        "fresh": ["fresh", "citrus", "beach", "ocean", "clean", "breezy", "lemon", "bergamot"],
        "floral": ["floral", "rose", "jasmine", "lily", "tuberose", "flower"],
        "woody": ["woody", "wood", "cedar", "sandalwood", "vetiver", "earthy"],
        "oud": ["oud", "habibi", "agarwood"],
        "sweet": ["sweet", "honey", "vanilla", "caramel", "cookie", "peaches", "gourmand", "dessert"],
        "spicy": ["spicy", "cinnamon", "cardamom", "nutmeg", "saffron", "pepper", "warm"],
        "oriental": ["oriental", "amber", "musk", "exotic"],
        "fruity": ["fruity", "fruit", "peach", "berry", "apple", "citrus"],
    }
    
    detected_scents = []
    for scent_type, keywords in scent_keywords.items():
        if any(kw in text_lower for kw in keywords):
            detected_scents.append(scent_type)
    return detected_scents


def _detect_budget(text: str) -> Optional[Dict[str, Any]]:
    """Detect budget preference from user text"""
    text_lower = text.lower()
    
    # Try to extract numeric budget values
    import re
    # Match patterns like "above 500", "over 600", "500+", "under 800", etc.
    numeric_patterns = [
        (r'(?:above|over|more than)\s*(\d+)', "over"),  # above 500
        (r'(\d+)\s*(?:\+|plus)', "over"),  # 500+
        (r'(?:under|below|less than)\s*(\d+)', "under"),  # under 800
        (r'around\s*(\d+)', "around"),  # around 700
        (r'\$?\s*(\d+)', "around"),  # $500, 500 (default to around if no operator)
    ]
    
    for pattern, default_op in numeric_patterns:
        match = re.search(pattern, text_lower)
        if match:
            amount = int(match.group(1))
            operator = default_op
            
            # Refine operator based on context if it was a generic number match
            if default_op == "around":
                if 'above' in text_lower or 'over' in text_lower or 'more than' in text_lower or '+' in text_lower:
                    operator = "over"
                elif 'under' in text_lower or 'below' in text_lower or 'less than' in text_lower:
                    operator = "under"
            
            return {"amount": amount, "operator": operator}
    
    # Fallback to keyword-based detection (map to approximate values)
    if any(word in text_lower for word in ["cheap", "budget", "affordable", "trial"]):
        return {"amount": 500, "operator": "under"}
    if any(word in text_lower for word in ["mid", "moderate"]):
        return {"amount": 800, "operator": "under"}
    if any(word in text_lower for word in ["premium", "high end", "luxury", "expensive", "best", "top"]):
        return {"amount": 1000, "operator": "over"}
        
    return None


def _detect_occasion(text: str) -> List[str]:
    """Detect occasion preferences from user text - returns list of matching occasions"""
    text_lower = text.lower()
    occasion_keywords = {
        "date_night": ["date", "romantic", "date night"],
        "evening": ["evening", "night out"],
        "formal": ["formal", "business", "meeting", "boardroom", "office", "work", "professional"],
        "casual": ["casual", "everyday", "daily", "daytime"],
        "party": ["party", "clubbing", "night", "celebration"],
        "special": ["wedding", "gala", "special occasion", "event"],
        "intimate": ["intimate", "sensual", "bedroom"],
    }
    
    detected_occasions = []
    for occasion, keywords in occasion_keywords.items():
        if any(kw in text_lower for kw in keywords):
            detected_occasions.append(occasion)
    return detected_occasions


def _detect_vibe(text: str) -> List[str]:
    """Detect vibe/mood preferences from user text"""
    text_lower = text.lower()
    vibe_keywords = {
        "bold": ["bold", "strong", "powerful", "confident"],
        "elegant": ["elegant", "sophisticated", "classy", "refined"],
        "playful": ["playful", "fun", "flirty", "wild"],
        "mature": ["mature", "serious", "professional"],
        "romantic": ["romantic", "passionate", "love"],
        "fresh": ["fresh", "clean", "light"],
        "warm": ["warm", "cozy", "comforting"],
    }
    
    detected_vibes = []
    for vibe, keywords in vibe_keywords.items():
        if any(kw in text_lower for kw in keywords):
            detected_vibes.append(vibe)
    return detected_vibes


def _detect_gift_intent(text: str) -> bool:
    """Detect if user is asking for gift packages, collections, or sets"""
    text_lower = text.lower()
    gift_keywords = [
        "gift", "gifting", "gift set", "gift package", "gift box", "present",
        "i want package", "want package", "show me package", "give me package",
        "i want collection", "want collection", "show me collection",
        "i want set", "want set", "show me set"
    ]
    # Exclude cases where user explicitly says they don't want these
    exclude_keywords = ["not a", "no ", "not ", "don't want", "dont want", "single"]
    has_exclude = any(excl in text_lower for excl in exclude_keywords)
    has_gift = any(kw in text_lower for kw in gift_keywords)
    return has_gift and not has_exclude


def _detect_single_perfume_intent(text: str) -> bool:
    """Detect if user wants a single perfume/bottle (not a gift set/package)"""
    text_lower = text.lower()
    single_keywords = [
        "single perfume", "single bottle", "single product", "one perfume", "one bottle",
        "a perfume", "a bottle", "individual perfume", "individual product", "just one",
        "not a set", "not a package", "not a gift set", "not package", "no package",
        "not a collection", "no collection", "not collection",
        "i want single", "want single", "need single"
    ]
    return any(kw in text_lower for kw in single_keywords)


def _detect_specific_notes(text: str) -> List[str]:
    """
    Detect specific perfume notes mentioned in user text.
    Uses dynamic vocabulary built from product catalog.
    Returns list of normalized note names found.
    Prioritizes longer/more specific matches (e.g., "vanilla absolute" over "vanilla").
    Also handles note position indicators like "rose heart" (rose in heart notes).
    """
    text_lower = text.lower()
    note_vocab = _get_note_vocabulary()
    detected_notes = []
    
    # Check for note position indicators (head, heart, base)
    note_positions = {"head": "head", "heart": "heart", "base": "base", "top": "head", "middle": "heart", "bottom": "base"}
    detected_position = None
    for pos_keyword, pos_name in note_positions.items():
        if pos_keyword in text_lower:
            detected_position = pos_name
            break
    
    # Sort keywords by length (longest first) to prioritize specific matches
    # This ensures "vanilla absolute" is checked before "vanilla"
    sorted_keywords = sorted(note_vocab.items(), key=lambda x: len(x[0]), reverse=True)
    
    # Track matched text to avoid duplicate processing
    matched_text = set()
    
    for note_keyword, note_variants in sorted_keywords:
        # Check if the keyword appears in the text
        if note_keyword in text_lower:
            # Check if a longer keyword already matched this part
            # (e.g., if "vanilla absolute" matched, don't also match "vanilla")
            is_substring_of_matched = any(
                note_keyword in matched and len(note_keyword) < len(matched)
                for matched in matched_text
            )
            
            if not is_substring_of_matched:
                # Add all variants of this note
                for variant in note_variants:
                    variant_lower = variant.lower()
                    if variant_lower not in detected_notes:
                        detected_notes.append(variant_lower)
                matched_text.add(note_keyword)
    
    # If a position was detected, store it in the note name for later filtering
    # We'll handle position filtering in the scoring function
    if detected_position and detected_notes:
        # Store position info by prefixing notes (we'll parse this in scoring)
        # For now, just ensure the notes are detected - position filtering happens in scoring
        pass
    
    return detected_notes


def extract_preferences_from_text(
    user_text: str, existing: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Merge new signals from the latest user message into the existing preference dict.
    """
    text = user_text.lower()
    prefs = existing.copy() if isinstance(existing, dict) else {}

    # Extract gender
    gender = _detect_gender(text)
    if gender:
        prefs["gender"] = gender

    # Extract scent types (can be multiple)
    scent_types = _detect_scent_type(text)
    if scent_types:
        existing_scents = prefs.get("scent_types", [])
        prefs["scent_types"] = list(set(existing_scents + scent_types))

    # Extract budget
    budget = _detect_budget(text)
    if budget:
        prefs["budget"] = budget

    # Extract occasions (can be multiple)
    occasions = _detect_occasion(text)
    if occasions:
        existing_occasions = prefs.get("occasions", [])
        prefs["occasions"] = list(set(existing_occasions + occasions))

    # Extract vibes (can be multiple)
    vibes = _detect_vibe(text)
    if vibes:
        existing_vibes = prefs.get("vibes", [])
        prefs["vibes"] = list(set(existing_vibes + vibes))

    # Extract gift intent - gift keywords take precedence over single perfume keywords
    # (e.g., "I want to gift a perfume" should be detected as gift, not single perfume)
    if _detect_gift_intent(text):
        prefs["is_gift"] = True
    elif _detect_single_perfume_intent(text):
        # User explicitly wants a single perfume, not a gift set
        prefs["is_gift"] = False

    # Extract specific notes (vanilla, rose, etc.)
    specific_notes = _detect_specific_notes(text)
    if specific_notes:
        existing_notes = prefs.get("specific_notes", [])
        prefs["specific_notes"] = list(set(existing_notes + specific_notes))

    return prefs


# ---------------------------------------------------------------------------
# 3. Enhanced Scoring engine
# ---------------------------------------------------------------------------

def _score_gender(pref_gender: Optional[str], perfume_gender: str) -> int:
    """Score based on gender matching"""
    if not pref_gender:
        return 0
    if pref_gender == perfume_gender:
        return 15  # Perfect match
    if perfume_gender == "unisex":
        return 8  # Unisex works for all
    return 0


def _score_specific_notes(pref_notes: List[str], perfume_notes: Dict[str, List[str]], user_text: str = "") -> int:
    """
    Score based on direct note matching.
    Higher score for exact matches, partial matches get lower score.
    Also considers note position (head/heart/base) if mentioned in user text.
    """
    if not pref_notes or not perfume_notes:
        return 0
    
    score = 0
    user_text_lower = user_text.lower() if user_text else ""
    
    # Check if user mentioned a specific note position
    note_positions = {"head": "head", "heart": "heart", "base": "base", "top": "head", "middle": "heart", "bottom": "base"}
    requested_position = None
    for pos_keyword, pos_name in note_positions.items():
        if pos_keyword in user_text_lower:
            requested_position = pos_name
            break
    
    # Get notes by position
    head_notes = [note.lower() for note in perfume_notes.get("head", [])]
    heart_notes = [note.lower() for note in perfume_notes.get("heart", [])]
    base_notes = [note.lower() for note in perfume_notes.get("base", [])]
    all_perfume_notes_lower = head_notes + heart_notes + base_notes
    
    for pref_note in pref_notes:
        pref_note_lower = pref_note.lower()
        
        # If user specified a position, prioritize matches in that position
        if requested_position:
            position_notes = {
                "head": head_notes,
                "heart": heart_notes,
                "base": base_notes
            }.get(requested_position, all_perfume_notes_lower)
            
            # Check exact match in requested position (highest score)
            if pref_note_lower in position_notes:
                score += 20  # Higher score for position-specific match
                continue
            # Check partial match in requested position
            for perfume_note_lower in position_notes:
                if pref_note_lower in perfume_note_lower or perfume_note_lower in pref_note_lower:
                    score += 15  # Good score for position-specific partial match
                    break
            # Also check other positions but with lower score
            other_positions_notes = [n for pos, notes in {
                "head": head_notes,
                "heart": heart_notes,
                "base": base_notes
            }.items() if pos != requested_position for n in notes]
            if pref_note_lower in other_positions_notes:
                score += 10  # Lower score for match in different position
            else:
                # Partial match in other positions
                for perfume_note_lower in other_positions_notes:
                    if pref_note_lower in perfume_note_lower or perfume_note_lower in pref_note_lower:
                        score += 7
                        break
        else:
            # No position specified - check all notes
            # Exact match (highest score)
            if pref_note_lower in all_perfume_notes_lower:
                score += 15
            else:
                # Partial match - check if pref_note is contained in any perfume note
                for perfume_note_lower in all_perfume_notes_lower:
                    if pref_note_lower in perfume_note_lower or perfume_note_lower in pref_note_lower:
                        score += 10
                        break  # Only count once per preference note
    
    return score


def _score_scent_types(pref_scents: List[str], perfume_vibes: List[str], perfume_notes: Dict[str, List[str]]) -> int:
    """
    Score based on scent type matching with vibe_tags and notes.
    Uses dynamic keyword mapping from scent_keywords instead of hardcoded values.
    """
    if not pref_scents:
        return 0
    
    score = 0
    # Check vibe_tags for scent matches
    for pref_scent in pref_scents:
        if pref_scent in perfume_vibes:
            score += 10
    
    # Get all notes from perfume
    all_notes = []
    if perfume_notes:
        all_notes = (perfume_notes.get("head", []) + 
                    perfume_notes.get("heart", []) + 
                    perfume_notes.get("base", []))
    
    all_notes_lower = [note.lower() for note in all_notes]
    
    # Use dynamic scent_keywords mapping instead of hardcoded checks
    scent_keywords = {
        "fresh": ["fresh", "citrus", "beach", "ocean", "clean", "breezy", "lemon", "bergamot", "orange", "passionfruit"],
        "floral": ["floral", "rose", "jasmine", "lily", "tuberose", "flower", "osmanthus"],
        "woody": ["woody", "wood", "cedar", "sandalwood", "vetiver", "earthy", "patchouli"],
        "oud": ["oud", "habibi", "agarwood"],
        "sweet": ["sweet", "honey", "vanilla", "caramel", "cookie", "peaches", "gourmand", "dessert", "tonka"],
        "spicy": ["spicy", "cinnamon", "cardamom", "nutmeg", "saffron", "pepper", "warm"],
        "oriental": ["oriental", "amber", "musk", "exotic"],
        "fruity": ["fruity", "fruit", "peach", "berry", "apple", "citrus"],
    }
    
    # Check if any keywords from the scent type appear in notes
    for pref_scent in pref_scents:
        keywords = scent_keywords.get(pref_scent, [])
        if keywords:
            # Check if any keyword from this scent type appears in notes
            matching_keywords = [kw for kw in keywords if any(kw in note for note in all_notes_lower)]
            if matching_keywords:
                score += 6  # Bonus for note keyword match
    
    return score


def _score_budget(pref_budget: Union[str, Dict[str, Any], None], price: int) -> int:
    """
    Score based on budget matching.
    Supports both old string format ("low", "mid", "high") and new dict format ({"amount": 500, "operator": "under"}).
    """
    if not pref_budget:
        return 0
        
    # Handle new dictionary format (Precise Budgeting)
    if isinstance(pref_budget, dict):
        amount = pref_budget.get("amount", 0)
        operator = pref_budget.get("operator", "around")
        
        if operator == "under":
            if price <= amount:
                return 20  # Fits budget
            else:
                return -1000  # STRICT PENALTY: Exceeds budget
        elif operator == "over":
            if price >= amount:
                return 20
            else:
                return -1000 # STRICT PENALTY: Below budget (unlikely to be an issue, but consistent)
        elif operator == "around":
            # Within 20% range
            if amount * 0.8 <= price <= amount * 1.2:
                return 20
            else:
                return 0
                
    # Handle old string format (Legacy/Fallback)
    if pref_budget == "low":
        return 10 if price <= 500 else (5 if price <= 650 else -1000) # Penalize expensive items for low budget
    if pref_budget == "mid":
        return 10 if 500 <= price <= 800 else (3 if price <= 900 else 0)
    if pref_budget == "high":
        return 10 if price >= 750 else 0
    return 0


def _score_occasions(pref_occasions: List[str], perfume_occasion_tags: List[str]) -> int:
    """Score based on occasion matching"""
    if not pref_occasions:
        return 0
    
    score = 0
    for pref_occ in pref_occasions:
        for perf_occ in perfume_occasion_tags:
            # Exact match
            if pref_occ == perf_occ:
                score += 8
            # Partial matches
            elif pref_occ in perf_occ or perf_occ in pref_occ:
                score += 5
    
    return score


def _score_vibes(pref_vibes: List[str], perfume_vibe_tags: List[str]) -> int:
    """Score based on vibe/mood matching"""
    if not pref_vibes:
        return 0
    
    score = 0
    for pref_vibe in pref_vibes:
        if pref_vibe in perfume_vibe_tags:
            score += 7
    
    return score


def score_perfume(perfume: Dict[str, Any], prefs: Dict[str, Any], boost_prefs: Optional[Dict[str, Any]] = None, user_text: str = "") -> int:
    """Calculate total score for a perfume based on user preferences"""
    score = 0
    
    # Gift intent matching (highest priority when gift is mentioned)
    if prefs.get("is_gift"):
        collection_tags = perfume.get("collection_tags", [])
        if "gift_set" in collection_tags:
            score += 50  # High boost for gift sets when gift is requested
        else:
            score -= 20  # Penalize non-gift products when gift is requested
    
    # Gender matching (highest priority)
    score += _score_gender(prefs.get("gender"), perfume.get("gender_profile", "unisex"))
    
    # Specific notes matching (high priority - direct note mentions like "vanilla", "rose")
    specific_notes = prefs.get("specific_notes", [])
    if specific_notes:
        score += _score_specific_notes(specific_notes, perfume.get("notes", {}), user_text)
    
    # Scent type matching
    score += _score_scent_types(
        prefs.get("scent_types", []), 
        perfume.get("vibe_tags", []),
        perfume.get("notes", {})
    )
    
    # Budget matching
    score += _score_budget(prefs.get("budget"), perfume.get("price", 0))
    
    # Occasion matching
    score += _score_occasions(
        prefs.get("occasions", []),
        perfume.get("occasion_tags", [])
    )
    
    # Vibe matching
    score += _score_vibes(
        prefs.get("vibes", []),
        perfume.get("vibe_tags", [])
    )
    
    # Bonus for best sellers
    if "best_seller" in perfume.get("collection_tags", []):
        score += 3
        
    # --- BOOST FOR LATEST PREFERENCES ---
    # If the user JUST mentioned something, it should override historical preferences
    if boost_prefs:
        # Boost for latest specific notes
        latest_notes = boost_prefs.get("specific_notes", [])
        if latest_notes:
            # Massive boost if the product matches the LATEST requested note
            # This ensures "anything with lily" immediately surfaces lily perfumes
            boost_score = _score_specific_notes(latest_notes, perfume.get("notes", {}), user_text)
            if boost_score > 0:
                score += 100  # MASSIVE BONUS
                
        # Boost for latest scent types
        latest_scents = boost_prefs.get("scent_types", [])
        if latest_scents:
            boost_score = _score_scent_types(latest_scents, perfume.get("vibe_tags", []), perfume.get("notes", {}))
            if boost_score > 0:
                score += 50
                
        # Boost for latest gender change
        latest_gender = boost_prefs.get("gender")
        if latest_gender:
             boost_score = _score_gender(latest_gender, perfume.get("gender_profile", "unisex"))
             if boost_score > 0:
                 score += 50
    
    return score


# ---------------------------------------------------------------------------
# 3.5. RAG Integration
# ---------------------------------------------------------------------------

def _rag_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Use the RAG service to find products semantically.
    """
    try:
        # Lazy import to avoid circular dependencies or startup issues
        from services.rag import get_rag_service
        rag = get_rag_service()
        
        # Initialize if needed (check if products are loaded)
        if not rag.products:
            # Try to initialize with the correct path
            from pathlib import Path
            products_path = Path(__file__).parent / "data" / "products.json"
            if products_path.exists():
                rag.initialize(str(products_path))
            else:
                return []
                
        results = rag.search(query, top_k=top_k)
        return results
    except Exception as e:
        print(f"RAG search failed: {e}")
        return []


# ---------------------------------------------------------------------------
# 4. Public function used by graph.py
# ---------------------------------------------------------------------------

def extract_preferences_and_recommend(
    user_text: str,
    existing_preferences: Optional[Dict[str, Any]] = None,
    limit: int = 2,  # Changed to 2 for better UX
    latest_user_text: Optional[str] = None,
    recently_recommended_ids: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Main entrypoint for your graph.

    1. Parse the latest user text into structured preferences
    2. Merge with existing preferences from state
    3. Score the catalog (Rule-based + RAG Hybrid)
    4. Return (updated_prefs, top_k_recommendations)
    """
    # Extract ONLY new preferences from the latest user text (for boosting)
    # If latest_user_text is provided, use it. Otherwise fallback to user_text.
    boost_text = latest_user_text if latest_user_text is not None else user_text
    new_prefs = extract_preferences_from_text(boost_text, existing=None)
    
    # Merge with existing preferences using the FULL context (user_text)
    # This ensures we don't lose context if we only look at the latest message for boosting
    prefs = extract_preferences_from_text(user_text, existing_preferences)

    # --- HYBRID SEARCH STRATEGY ---
    
    # 1. Get RAG candidates (Semantic Search)
    # Use the boost_text (latest query) for RAG to capture immediate intent
    rag_candidates = _rag_search(boost_text, top_k=10)
    rag_ids = [p.get("id") for p in rag_candidates]
    
    scored: List[Tuple[int, Dict[str, Any]]] = []
    recent_ids = recently_recommended_ids or []
    
    for item in PERFUME_CATALOG:
        # Base Rule-based Score
        # Use boost_text (latest query) for note position detection (e.g., "rose heart")
        s = score_perfume(item, prefs, boost_prefs=new_prefs, user_text=boost_text)
        
        # RAG Boost: If item is in RAG results, give it a boost based on rank
        if item.get("id") in rag_ids:
            rank = rag_ids.index(item.get("id"))
            # Boost top RAG results: +40 for 1st, +35 for 2nd, etc.
            rag_boost = max(0, 40 - (rank * 5))
            s += rag_boost
        
        # Apply penalty for recently recommended products to encourage variety
        if item.get("id") in recent_ids:
            s -= 30  # Significant but not insurmountable penalty
            
        scored.append((s, item))

    # sort by score desc
    scored.sort(key=lambda x: x[0], reverse=True)

    # take top-k, but only those with positive score
    # Smart gift filtering: if user has specific preferences (notes, scent types), show individual perfumes
    # Only strictly filter for gift sets if is_gift=True AND no specific preferences
    is_gift = prefs.get("is_gift", False)
    has_specific_prefs = (
        len(prefs.get("specific_notes", [])) > 0 or
        len(prefs.get("scent_types", [])) > 0
    )
    strict_gift_filter = is_gift and not has_specific_prefs
    
    top_items: List[Dict[str, Any]] = []
    candidates = scored[:limit * 3] if is_gift else scored[:limit * 2]  # Look at more candidates
    
    for score, item in candidates:
        if score <= 0:
            continue
        
        # Apply strict gift filtering only if user hasn't added specific preferences
        # This allows "I want a gift" -> shows gift sets
        # But "I want a gift, she likes rose" -> shows individual rose perfumes
        if strict_gift_filter and "gift_set" not in item.get("collection_tags", []):
            continue
        
        # If is_gift=False (user said "single perfume"), exclude gift sets
        if is_gift == False and "gift_set" in item.get("collection_tags", []):
            continue
            
        # project to a response-friendly shape with all necessary fields
        top_items.append(
            {
                "id": item.get("id", ""),
                "name": item.get("name", ""),
                "score": score,
                "gender_profile": item.get("gender_profile", "unisex"),
                "price": item.get("price", 0),
                "mrp": item.get("mrp", 0),
                "short_description": item.get("short_description", ""),
                "positioning_line": item.get("positioning_line", ""),
                "vibe_tags": item.get("vibe_tags", []),
                "occasion_tags": item.get("occasion_tags", []),
                "notes": item.get("notes", {}),
                "image_url": item.get("image_url", ""),
                "product_url": item.get("product_url", ""),
                "reviews_count": item.get("reviews_count", 0),
            }
        )
        
        if len(top_items) >= limit:
            break

    return prefs, top_items
