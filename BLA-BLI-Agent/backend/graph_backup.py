import os
from typing import TypedDict, List, Optional, Dict, Any
import re
import logging
import time

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from openai import OpenAI

from prompts import INTENT_PROMPT, ASSISTANT_STYLE_PROMPT, PREFERENCE_EXTRACTION_PROMPT
from tools import (
    get_product_by_name,
    list_products,
    recommend_products,
    get_order_status,
    get_return_policy,
    get_byob_products,
    validate_byob_selection,
    get_faq_answer,
)

try:
    from recommender import extract_preferences_and_recommend
except Exception:
    extract_preferences_and_recommend = None

logger = logging.getLogger("blabli.graph")
_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger.setLevel(_level)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
    logger.addHandler(_h)
logger.propagate = False


def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Put it in .env at project root.")
    return OpenAI(api_key=api_key)


class ChatState(TypedDict, total=False):
    messages: List[dict]
    intent: Optional[str]
    customer_authed: bool
    customer_email: Optional[str]
    order_number: Optional[str]
    tool_result: Optional[dict]
    missing_field: Optional[str]
    preferences: Optional[Dict[str, Any]]
    recommendations: Optional[List[Dict[str, Any]]]
    last_recommendations: Optional[List[Dict[str, Any]]]  # Track last recommended products for follow-up questions
    recommendation_index: int  # Track which recommendation we've shown (0-indexed)
    byob_selections: Optional[List[str]]  # Track BYOB product selections (IDs)
    byob_max_items: int  # Max items allowed in BYOB (default 3)
    button_suggestions: Optional[List[Dict[str, str]]]  # Button suggestions to show to user
    awaiting_choice: Optional[str]  # Track what choice we're waiting for (vibe, occasion, gender, etc.)
    questions_asked: Optional[List[str]]  # Track which questions have been asked (gender, vibe, occasion, budget)
    handoff_state: Optional[str]  # Track human handoff flow: None, "awaiting_confirmation", "collecting_email", "collecting_phone", "completed"
    handoff_email: Optional[str]  # Email collected for handoff
    handoff_phone: Optional[str]  # Phone collected for handoff
    handoff_completed: bool  # Flag to stop bot after handoff


def _call_llm(prompt: str) -> str:
    """
    Wrapper that works with both the modern OpenAI Responses API (>=1.0)
    and the legacy Chat Completions API (<1.0).
    """
    client = get_client()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if hasattr(client, "responses"):
        resp = client.responses.create(model=model, input=prompt)
        return resp.output[0].content[0].text.strip()

    # fallback for older openai package versions
    if not hasattr(client, "chat") or not hasattr(client.chat, "completions"):
        raise RuntimeError("OpenAI client does not support responses or chat completions APIs.")

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    message = resp.choices[0].message
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    return content.strip()


def _call_intent_llm(prompt: str) -> str:
    # Prompt is now constructed by the caller (intent_router) to include context
    return _call_llm(prompt)


def intent_router(state: ChatState):
    logger.info("node=intent_router entering")
    last_user_msg = ""
    last_assistant_msg = ""
    
    # Find last user message and last assistant message
    for m in reversed(state["messages"]):
        if m["role"] == "user" and not last_user_msg:
            last_user_msg = m["content"]
        elif m["role"] == "assistant" and not last_assistant_msg:
            last_assistant_msg = m["content"]
        
        if last_user_msg and last_assistant_msg:
            break
    
    # Direct intent mapping for button actions (bypassing LLM for efficiency)
    user_msg_lower = last_user_msg.lower()
    
    # Handle handoff button actions
    if "confirm_handoff" in last_user_msg or user_msg_lower in ["yes", "yeah", "yep", "sure", "ok", "okay", "connect me"]:
        handoff_state = state.get("handoff_state")
        if handoff_state == "awaiting_confirmation":
            state["intent"] = "human_handoff"
            logger.info("node=intent_router handoff_confirmed=1")
            return state
    
    # Special handling: If user is answering a direct choice question with a button-like response
    # Only continue intent if they're literally answering the question we asked
    awaiting_choice = state.get("awaiting_choice")
    if awaiting_choice:
        last_intent = state.get("intent")
        # Check if this is a button-style answer to the question we asked
        is_button_answer = False
        
        if awaiting_choice == "byob_selection":
            # If we're awaiting BYOB selection, check if user is actually selecting a product
            # or asking a different question (like "when to wear", "what is the price")
            # Check for non-BYOB question patterns that should break out of BYOB flow
            non_byob_patterns = [
                "what is", "what's", "how much", "tell me", "when", "wear", "occasion",
                "i want to buy", "i want a", "find", "looking for", "recommend", "suggest", 
                "gift", "track", "order", "notes", "what do i get", "contents"
            ]
            is_non_byob_question = any(pattern in user_msg_lower for pattern in non_byob_patterns)
            
            # Only continue BYOB intent if it looks like a product selection, not a general question
            if last_intent == "build_own_box" and not is_non_byob_question:
                logger.info("node=intent_router continuing_intent=build_own_box awaiting=byob_selection")
                state["intent"] = "build_own_box"
                return state
            elif is_non_byob_question:
                # User is asking a different question, clear awaiting_choice and re-detect intent
                logger.info("node=intent_router breaking_out_of_byob user_asked_different_question=1")
                state["awaiting_choice"] = None
                # Don't return here, let the intent be re-detected below
        elif awaiting_choice == "gender" and any(word in user_msg_lower for word in ["for him", "for her", "unisex", "men's", "women's"]):
            is_button_answer = True
        elif awaiting_choice == "vibe" and any(word in user_msg_lower for word in ["fresh", "sweet", "spicy", "oud", "floral", "woody", "breezy", "gourmand", "cozy", "earthy"]):
            is_button_answer = True
        elif awaiting_choice == "occasion" and any(word in user_msg_lower for word in ["daily", "office", "date night", "party", "mixed", "casual", "formal"]):
            is_button_answer = True
        elif awaiting_choice == "budget" and any(word in user_msg_lower for word in ["under", "above", "over", "700", "900", "500", "no limit", "any", "₹"]):
            is_button_answer = True
        
        if is_button_answer and last_intent in ("find_perfume", "product_recommendation", "gift_recommendation"):
            logger.info("node=intent_router continuing_intent=%s awaiting=%s detected_button_answer=1", last_intent, awaiting_choice)
            state["intent"] = last_intent
            return state
    
    # Also check if user is changing preferences after "no recommendations" scenario
    # Detect button actions like "Above ₹500", "Office/Formal", etc.
    if any(phrase in user_msg_lower for phrase in ["above ₹", "under ₹", "office/formal", "daily/casual"]):
        last_intent = state.get("intent")
        if last_intent in ("find_perfume", "product_recommendation", "gift_recommendation"):
            logger.info("node=intent_router detected_preference_change continuing_intent=%s", last_intent)
            state["intent"] = last_intent
            # Clear awaiting_choice so preferences are re-extracted
            state["awaiting_choice"] = None
            return state
    
    direct_intent_map = {
        "find my perfume": "find_perfume",
        "gift for someone": "gift_recommendation",
        "build my own box": "build_own_box",
        "track my order": "order_status",
        "looking for perfume": "find_perfume",
        "want a gift": "gift_recommendation",
        "create my box": "build_own_box",
        "track order": "order_status",
        "talk to human": "human_handoff",
        "speak with human": "human_handoff",
        "talk to agent": "human_handoff",
        "speak with agent": "human_handoff",
        "talk to representative": "human_handoff",
        "speak with representative": "human_handoff",
        "talk to support": "human_handoff",
        "speak with support": "human_handoff",
        "human agent": "human_handoff",
        "connect me to human": "human_handoff",
    }
    
    for phrase, mapped_intent in direct_intent_map.items():
        if phrase in user_msg_lower:
            state["intent"] = mapped_intent
            logger.info("node=intent_router direct_map_intent=%s", mapped_intent)
            return state
    
    # Check if this is a product question about a specific product (with or without last recommendations)
    has_last_recs = bool(state.get("last_recommendations"))
    
    # Common product question patterns
    product_question_patterns = [
        "what is the price", "what is its price", "how much", "what's the price",
        "what do i get", "what's in it", "what comes with", "contents",
        "when can i wear", "when to wear", "when should i wear", "suitable for",
        "what occasion", "what notes", "what nodes", "notes in", "nodes in",
        "fragrance notes", "tell me more", "more about",
    ]
    
    # Check if user is asking about product details
    is_product_question = any(pattern in user_msg_lower for pattern in product_question_patterns)
    
    # Also check if user mentions a specific product name (even without last_recs)
    # This handles cases like "what notes are in love drunk" when product wasn't recommended
    has_product_name_mention = any(word.istitle() or len(word) > 4 for word in last_user_msg.split() 
                                     if word.lower() not in ["what", "notes", "nodes", "price", "present", "about"])
    
    if is_product_question and (has_last_recs or has_product_name_mention):
        state["intent"] = "product_question"
        logger.info("node=intent_router direct_map_product_question=1 has_last_recs=%s has_name=%s", 
                   has_last_recs, has_product_name_mention)
        return state
    
    # Check if user is asking about previously recommended products
    # This helps improve intent detection for product_question
    
    # Construct prompt with context
    prompt = INTENT_PROMPT
    
    # Pass current intent and state to help LLM preserve context
    current_intent = state.get("intent")
    awaiting = state.get("awaiting_choice")
    if current_intent:
        prompt += f"\n\nCurrent Intent: {current_intent}"
    if awaiting:
        prompt += f"\nCurrently asking user for: {awaiting}"
    
    if last_assistant_msg:
        prompt += f"\nAssistant's last message: {last_assistant_msg}"
    
    if has_last_recs:
        # Add context about recommended products to help intent detection
        last_recs = state.get("last_recommendations", [])
        product_names = [p.get("name", "") for p in last_recs[:3]]  # Show up to 3 products
        prompt += f"\nContext: Products just recommended: {', '.join(product_names)}"
        
    prompt += f"\nUser: {last_user_msg}"
    
    intent = _call_intent_llm(prompt)
    
    state["intent"] = intent
    logger.info("node=intent_router detected_intent=%s has_last_recs=%s", intent, has_last_recs)
    return state


def auth_guard(state: ChatState):
    logger.info(
        "node=auth_guard intent=%s email_present=%s order_present=%s",
        state.get("intent"),
        bool(state.get("customer_email")),
        bool(state.get("order_number")),
    )
    # If we're in an auth-requiring flow, try to extract missing fields from latest user message
    requires_auth = state["intent"] in ("order_status", "change_address") or state.get("missing_field") == "auth"

    if requires_auth:
        last_user_msg = ""
        for m in reversed(state["messages"]):
            if m["role"] == "user":
                last_user_msg = m["content"]
                break

        # Fill email if missing
        if not state.get("customer_email") and last_user_msg:
            email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", last_user_msg)
            if email_match:
                state["customer_email"] = email_match.group(0)
                logger.info("node=auth_guard extracted_email=1")

        # Fill order number if missing (e.g., BLB-1001)
        if not state.get("order_number") and last_user_msg:
            order_match = re.search(r"\bBLB-\d{3,}\b", last_user_msg, flags=re.IGNORECASE)
            if order_match:
                state["order_number"] = order_match.group(0).upper()
                logger.info("node=auth_guard extracted_order_number=1")

        if not state.get("customer_email") or not state.get("order_number"):
            state["missing_field"] = "auth"
            state["customer_authed"] = False
            logger.info("node=auth_guard auth_status=missing")
        else:
            state["missing_field"] = None
            state["customer_authed"] = True
            logger.info("node=auth_guard auth_status=ok")
    else:
        state["missing_field"] = None
    return state


def collect_preferences(state: ChatState):
    """
    For product-related intents, normalize or initialize the user's preference profile
    before we hit tools / LLM. This makes the bot contextual across turns.
    """
    intent = state.get("intent")
    if intent not in ("product_info", "product_recommendation", "find_perfume", "gift_recommendation"):
        return state

    # Get all user messages from conversation history to extract preferences from context
    user_messages = [m["content"] for m in state["messages"] if m["role"] == "user"]
    last_user_msg = user_messages[-1] if user_messages else ""
    last_user_msg_lower = last_user_msg.lower()
    # Combine recent user messages for better context (last 3 messages)
    context_text = " ".join(user_messages[-3:]) if len(user_messages) > 0 else ""

    # ensure preferences dict exists (adapted for perfumes) - preserve existing preferences
    prefs = state.get("preferences") or {
        "gender": None,
        "scent_types": [],
        "budget": None,
        "occasions": [],
        "vibes": [],
        "specific_notes": [],
        "is_gift": False,
        "occasion_raw": None,  # Store raw user input for occasion (e.g., "wedding", "date night")
    }
    
    # Direct preference detection from button-like responses (faster than LLM)
    # This handles cases where user types instead of clicking buttons
    awaiting_choice = state.get("awaiting_choice")
    
    if awaiting_choice == "gender":
        # User was asked about gender or is updating gender
        # Allow updates even if gender is already set (user can change their mind)
        # Check for specific gender terms first (more specific patterns win)
        if any(word in last_user_msg_lower for word in ["men's", "mens", "for him", "male perfume", "man's", "guy's"]):
            prefs["gender"] = "for_him"
            logger.info("node=collect_preferences direct_detect_gender=for_him")
        elif any(word in last_user_msg_lower for word in ["women's", "womens", "for her", "female perfume", "woman's", "lady's"]):
            prefs["gender"] = "for_her"
            logger.info("node=collect_preferences direct_detect_gender=for_her")
        elif "unisex" in last_user_msg_lower or "both" in last_user_msg_lower or "either" in last_user_msg_lower:
            prefs["gender"] = "unisex"
            logger.info("node=collect_preferences direct_detect_gender=unisex")
        # If they just say generic words without specific gender indicator, try to detect
        elif any(word in last_user_msg_lower for word in ["men", "male", "him", "guy", "man"]) and "women" not in last_user_msg_lower:
            prefs["gender"] = "for_him"
            logger.info("node=collect_preferences direct_detect_gender=for_him (generic)")
        elif any(word in last_user_msg_lower for word in ["women", "female", "her", "lady", "woman"]) and "men" not in last_user_msg_lower:
            prefs["gender"] = "for_her"
            logger.info("node=collect_preferences direct_detect_gender=for_her (generic)")
    
    if awaiting_choice == "vibe":
        # User was asked about vibe/scent or is updating vibe
        # Allow updates even if vibe is already set (user can change their mind)
        vibe_map = {
            "fresh": ["fresh", "breezy", "citrus", "clean"],
            "sweet": ["sweet", "gourmand", "vanilla", "honey"],
            "spicy": ["spicy", "bold", "warm"],
            "oud": ["oud", "intense", "habibi"],
            "floral": ["floral", "soft", "cozy", "rose"],
            "woody": ["woody", "earthy", "wood"],
        }
        for vibe_type, keywords in vibe_map.items():
            if any(kw in last_user_msg_lower for kw in keywords):
                prefs["scent_types"] = [vibe_type]
                logger.info("node=collect_preferences direct_detect_vibe=%s", vibe_type)
                break
    
    if awaiting_choice == "occasion":
        # User was asked about occasion or is updating occasion
        # Allow updates even if occasion is already set (user can change their mind)
        occasion_map = {
            "casual": ["daily", "everyday", "casual", "vacation", "holiday", "trip"],
            "formal": ["office", "work", "formal", "business", "wedding", "ceremony", "function", "event"],
            "date_night": ["date", "romantic", "date night"],
            "party": ["party", "night", "clubbing"],
            "mixed": ["mixed", "all", "everything", "any"],
        }
        for occ_type, keywords in occasion_map.items():
            if any(kw in last_user_msg_lower for kw in keywords):
                prefs["occasions"] = [occ_type]
                # Store raw occasion text for natural reference later
                for keyword in keywords:
                    if keyword in last_user_msg_lower:
                        prefs["occasion_raw"] = keyword
                        break
                logger.info("node=collect_preferences direct_detect_occasion=%s raw=%s", occ_type, prefs.get("occasion_raw"))
                break
    
    # Handle budget updates (both when awaiting and when user spontaneously changes)
    if awaiting_choice == "budget" or any(word in last_user_msg_lower for word in ["budget", "above", "under", "over", "₹"]):
        # User was asked about budget or is updating budget preference
        # Allow updates even if budget is already set (user can change their mind)
        budget_updated = False
        
        if "under" in last_user_msg_lower and "700" in last_user_msg_lower:
            prefs["budget"] = {"amount": 700, "operator": "under"}
            logger.info("node=collect_preferences direct_detect_budget=under_700")
            budget_updated = True
        elif ("above" in last_user_msg_lower or "over" in last_user_msg_lower or "more than" in last_user_msg_lower):
            # Extract amount dynamically
            import re
            amount_match = re.search(r'(\d+)', last_user_msg_lower)
            if amount_match:
                amount = int(amount_match.group(1))
                prefs["budget"] = {"amount": amount, "operator": "over"}
                logger.info("node=collect_preferences direct_detect_budget=above_%d", amount)
                budget_updated = True
        elif "700" in last_user_msg_lower and "900" in last_user_msg_lower:
            prefs["budget"] = {"amount": 800, "operator": "around"}
            logger.info("node=collect_preferences direct_detect_budget=700_900")
            budget_updated = True
        elif "no limit" in last_user_msg_lower or ("any" in last_user_msg_lower and ("budget" in last_user_msg_lower or awaiting_choice == "budget")):
            prefs["budget"] = None  # No budget constraint
            logger.info("node=collect_preferences direct_detect_budget=no_limit")
            budget_updated = True
        
        # Mark budget as asked if it was updated
        if budget_updated and "budget" not in state.get("questions_asked", []):
            questions_asked = state.get("questions_asked", [])
            questions_asked.append("budget")
            state["questions_asked"] = questions_asked

    # Use LLM to extract preferences from context (Smart extraction)
    try:
        # Pass current preferences to LLM so it doesn't lose context
        current_prefs_str = f"""
Current preferences already set:
- Gender: {prefs.get('gender', 'not set')}
- Is Gift: {prefs.get('is_gift', False)}
- Scent Types: {prefs.get('scent_types', [])}
- Budget: {prefs.get('budget', 'not set')}
- Occasions: {prefs.get('occasions', [])}

IMPORTANT: Only update fields that are EXPLICITLY mentioned in the latest user message.
If user only mentions budget (e.g., "under 500"), keep all other fields unchanged.
Preserve is_gift and gender unless user explicitly changes them.
"""
        
        llm_prompt = f"""{PREFERENCE_EXTRACTION_PROMPT}

{current_prefs_str}

Conversation Context:
{context_text}

Latest User Message: {last_user_msg}
"""
        llm_response = _call_llm(llm_prompt)
        # Clean up response to ensure it's valid JSON
        llm_response = llm_response.strip()
        if llm_response.startswith("```json"):
            llm_response = llm_response[7:]
        if llm_response.endswith("```"):
            llm_response = llm_response[:-3]
        
        import json
        extracted_prefs = json.loads(llm_response)
        
        # Merge extracted prefs into existing prefs - PRESERVE CONTEXT
        # Only update fields that are explicitly mentioned in the latest message
        if extracted_prefs.get("gender") and extracted_prefs.get("gender") != prefs.get("gender"):
            # Only update if LLM detected a NEW gender (not just repeating existing)
            prefs["gender"] = extracted_prefs["gender"]
        
        if extracted_prefs.get("scent_types"):
            # If user is changing scent type (new message has scent), replace instead of merge
            # Check if this is a fresh scent selection or an addition
            if len(extracted_prefs["scent_types"]) > 0:
                # If user explicitly mentions a scent in latest message, it's likely a change
                # Replace instead of merge to avoid confusion
                prefs["scent_types"] = extracted_prefs["scent_types"]
            
        if extracted_prefs.get("budget"):
            # Always trust LLM for budget - if user mentions budget, they want to change it
            prefs["budget"] = extracted_prefs["budget"]
            logger.info("node=collect_preferences llm_budget_updated amount=%s operator=%s", 
                       extracted_prefs["budget"].get("amount"), extracted_prefs["budget"].get("operator"))
            
        if extracted_prefs.get("is_gift") is not None:
            # Only update is_gift if it's explicitly mentioned in latest message
            # If is_gift is already True (from previous context), preserve it unless user explicitly says otherwise
            if prefs.get("is_gift") is True:
                # Already a gift - only change if user explicitly says it's NOT a gift
                # (which is rare - usually they just mention budget/preferences)
                # So we preserve the gift status
                pass
            else:
                # Not yet marked as gift - trust LLM if it detects gift intent
                if extracted_prefs["is_gift"]:
                    prefs["is_gift"] = True
                    logger.info("node=collect_preferences llm_gift_detection is_gift=%s", extracted_prefs["is_gift"])

        if extracted_prefs.get("specific_notes"):
            existing_notes = prefs.get("specific_notes", [])
            new_notes = extracted_prefs["specific_notes"]
            prefs["specific_notes"] = list(set(existing_notes + new_notes))
            
        logger.info("node=collect_preferences llm_extraction_success=1 extracted=%s", extracted_prefs)
    except Exception as e:
        logger.error("node=collect_preferences llm_extraction_error=%s", e)

    # If we have a recommender module, let it do smarter extraction (Rule-based + Scoring)
    # Use context_text to extract preferences from entire conversation, not just last message
    if extract_preferences_and_recommend is not None:
        try:
            # Get recently recommended IDs to avoid repetition
            last_recs = state.get("last_recommendations") or []
            recent_ids = [p.get("id") for p in last_recs if isinstance(p, dict) and "id" in p]
            
            # Pass the updated prefs (with LLM extraction) to the recommender
            # Pass context_text for extraction, but last_user_msg for boosting
            rec_out = extract_preferences_and_recommend(
                context_text, 
                prefs, 
                latest_user_text=last_user_msg,
                recently_recommended_ids=recent_ids
            )
            # Support both tuple-style and dict-style return values
            if isinstance(rec_out, tuple):
                # recommender returned (new_prefs, recos)
                new_prefs, recos = rec_out
            else:
                # recommender returned {"preferences": {...}, "recommendations": [...]}
                new_prefs = rec_out.get("preferences") or prefs
                recos = rec_out.get("recommendations") or []
            # Update state
            state["preferences"] = new_prefs or prefs
            
            # Only set recommendations if we have sufficient preferences
            # For gift queries, need gender + vibe + budget (3 questions)
            # For regular queries, need ALL 4 questions asked
            is_gift = new_prefs.get("is_gift", False)
            questions_asked = state.get("questions_asked", [])
            required_questions = {"gender", "vibe", "occasion", "budget"}
            all_questions_asked = required_questions.issubset(set(questions_asked))
            
            if is_gift:
                gift_required_questions = {"gender", "vibe", "budget"}
                has_sufficient_prefs = gift_required_questions.issubset(set(questions_asked))
            else:
                has_sufficient_prefs = all_questions_asked
            
            if recos and has_sufficient_prefs:
                # pre-fill tool_result so tool_exec/response_node can use it directly
                state["tool_result"] = {"recommendations": recos}
            else:
                # Clear any pre-existing recommendations if we don't have enough preferences
                state["tool_result"] = None
            return state
        except Exception as e:
            logger.exception("node=collect_preferences error=%s", e)

    state["preferences"] = prefs
    return state


def tool_exec(state: ChatState):
    logger.info("node=tool_exec intent=%s", state.get("intent"))
    # if a previous node already put recommendations into tool_result, keep it
    if state.get("tool_result") and "recommendations" in state["tool_result"]:
        logger.info("node=tool_exec detected_pre_filled_recommendations=1")
        # Reset recommendation index when new recommendations are generated
        state["recommendation_index"] = 0
        return state

    intent = state["intent"]
    result = {}

    if state.get("missing_field") == "auth":
        state["tool_result"] = None
        logger.info("node=tool_exec blocked_missing_auth=1")
        return state

    if intent == "product_info":
        user_text = next(m["content"] for m in reversed(state["messages"]) if m["role"] == "user")
        prod = get_product_by_name(user_text)
        if prod:
            result = {"product": prod}
            logger.info("node=tool_exec path=product_info hit=exact")
        else:
            result = {"products": list_products()}
            logger.info("node=tool_exec path=product_info hit=list")
    elif intent in ("product_recommendation", "find_perfume", "gift_recommendation"):
        # If collect_preferences already set recommendations (with sufficient prefs), use them
        if state.get("tool_result") and isinstance(state.get("tool_result"), dict) and "recommendations" in state["tool_result"]:
            logger.info("node=tool_exec path=%s using_pre_filled_recommendations", intent)
            # Reset recommendation index when new recommendations are generated
            state["recommendation_index"] = 0
            return state
        
        # If collect_preferences set tool_result to None (insufficient prefs), preserve it
        if state.get("tool_result") is None:
            logger.info("node=tool_exec path=%s preserving_none_from_collect_prefs", intent)
            return state
        
        # Check if we have sufficient preferences before recommending
        prefs = state.get("preferences") or {}
        has_gender = prefs.get("gender") is not None
        has_scent_type = isinstance(prefs.get("scent_types"), list) and len(prefs.get("scent_types", [])) > 0
        has_budget = prefs.get("budget") is not None
        has_occasion = isinstance(prefs.get("occasions"), list) and len(prefs.get("occasions", [])) > 0
        is_gift = prefs.get("is_gift", False) or intent == "gift_recommendation"
        
        # Check if all required questions have been asked
        questions_asked = state.get("questions_asked", [])
        required_questions = {"gender", "vibe", "occasion", "budget"}
        all_questions_asked = required_questions.issubset(set(questions_asked))
        
        # For gift queries, need gender + vibe + budget (3 questions)
        # For regular queries, need ALL 4 questions asked
        if is_gift:
            gift_required_questions = {"gender", "vibe", "budget"}
            has_sufficient_prefs = gift_required_questions.issubset(set(questions_asked))
        else:
            has_sufficient_prefs = all_questions_asked
        
        # Only recommend if we have sufficient preferences
        if has_sufficient_prefs:
            user_text = next(m["content"] for m in reversed(state["messages"]) if m["role"] == "user")
            recs = recommend_products(user_text)
            result = {"recommendations": recs}
            # Reset recommendation index when new recommendations are generated
            state["recommendation_index"] = 0
            logger.info("node=tool_exec path=%s count=%d", intent, len(recs))
        else:
            # No sufficient preferences - don't set recommendations, response_node will ask questions
            result = None
            logger.info("node=tool_exec path=%s insufficient_prefs gender=%s scent=%s budget=%s is_gift=%s", 
                       intent, has_gender, has_scent_type, has_budget, is_gift)
    elif intent == "build_own_box":
        # Get BYOB products
        byob_products = get_byob_products()
        result = {"byob_products": byob_products}
        logger.info("node=tool_exec path=build_own_box count=%d", len(byob_products))
    elif intent == "general_question":
        user_text = next(m["content"] for m in reversed(state["messages"]) if m["role"] == "user")
        faq_answer = get_faq_answer(user_text)
        result = {"faq_answer": faq_answer}
        logger.info("node=tool_exec path=general_question found=%s", bool(faq_answer))
    elif intent == "returns_refunds":
        result = get_return_policy()
        logger.info("node=tool_exec path=returns_refunds")
    elif intent == "order_status":
        r = get_order_status(state["customer_email"], state["order_number"])
        result = r
        logger.info("node=tool_exec path=order_status error=%s", str(r.get("error")) if isinstance(r, dict) else "none")
    elif intent == "change_address":
        result = {"ok": False, "message": "Address change not allowed after dispatch."}
        logger.info("node=tool_exec path=change_address")
    else:
        result = {}
        logger.info("node=tool_exec path=unknown")

    state["tool_result"] = result
    return state


def response_node(state: ChatState):
    logger.info("node=response entering intent=%s missing=%s", state.get("intent"), state.get("missing_field"))
    user_text = next(m["content"] for m in reversed(state["messages"]) if m["role"] == "user")
    intent = state.get("intent")
    tool_result = state.get("tool_result")
    missing = state.get("missing_field")
    customer_authed = state.get("customer_authed", False)
    customer_email = state.get("customer_email")
    order_number = state.get("order_number")
    
    # Save awaiting_choice before clearing (needed for BYOB and preference flows)
    awaiting_choice_before_clear = state.get("awaiting_choice")
    
    # Clear button suggestions at start of response (will be regenerated if needed)
    state["button_suggestions"] = None
    # Don't clear awaiting_choice yet - we need it for BYOB and preference detection
    # It will be cleared or updated later in the flow

    # Handle human handoff flow
    handoff_state = state.get("handoff_state")
    if intent == "human_handoff" or handoff_state:
        # Force intent to human_handoff if we're in handoff flow
        if handoff_state:
            state["intent"] = "human_handoff"
        
        # Extract email/phone from user message if provided
        user_text_lower = user_text.lower()
        email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", user_text)
        phone_match = re.search(r"[\d\s\-\+\(\)]{10,}", user_text)
        
        if handoff_state == "collecting_email":
            # User is providing email
            if email_match:
                state["handoff_email"] = email_match.group(0)
                state["handoff_state"] = "collecting_phone"
                reply = "Thank you! Please provide your phone number so we can contact you."
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response handoff_email_collected=%s", state["handoff_email"])
                return state
            else:
                reply = "I didn't catch a valid email address. Please provide your email address."
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response handoff_email_invalid=1")
                return state
        elif handoff_state == "collecting_phone":
            # User is providing phone
            if phone_match:
                state["handoff_phone"] = phone_match.group(0).strip()
                # Generate ticket (in real implementation, this would call a ticket system API)
                ticket_id = f"TKT-{int(time.time())}"
                state["handoff_state"] = None  # Clear handoff state to allow normal conversation
                state["handoff_completed"] = True  # Keep flag for reference but don't block
                reply = f"Ok, we have raised the ticket (Ticket ID: {ticket_id}). Someone will contact you shortly at {state['handoff_email']} or {state['handoff_phone']}. Can I help you with anything else?"
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response handoff_completed ticket_id=%s email=%s phone=%s", 
                          ticket_id, state["handoff_email"], state["handoff_phone"])
                return state
            else:
                reply = "I didn't catch a valid phone number. Please provide your phone number."
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response handoff_phone_invalid=1")
                return state
        elif handoff_state == "awaiting_confirmation":
            # User is confirming they want to talk to human
            user_text_lower = user_text.lower()
            if any(word in user_text_lower for word in ["yes", "yeah", "yep", "sure", "ok", "okay", "confirm", "proceed", "connect me"]) or "confirm_handoff" in user_text:
                state["handoff_state"] = "collecting_email"
                reply = "Great! To connect you with a human agent, I'll need your email address first."
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response handoff_confirmed=1")
                return state
            elif any(word in user_text_lower for word in ["no", "nope", "cancel", "cancel_handoff"]) or "cancel_handoff" in user_text:
                # User declined or changed mind
                state["handoff_state"] = None
                reply = "No problem! How else can I help you today?"
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response handoff_declined=1")
                return state
        else:
            # First time - ask for confirmation
            state["handoff_state"] = "awaiting_confirmation"
            reply = "I understand you'd like to speak with a human agent. Would you like me to connect you with one?"
            buttons = [
                {"label": "Yes", "action": "confirm_handoff", "type": "primary"},
                {"label": "No", "action": "cancel_handoff", "type": "secondary"},
            ]
            state["button_suggestions"] = buttons
            state["messages"].append({"role": "assistant", "content": reply})
            logger.info("node=response handoff_initiated=1")
            return state

    if missing == "auth":
        reply = (
            "I can look up your order. Please share the email you used and the order number (like BLB-1001). "
            "After that, I'll tell you the exact status. Can I help you with anything else?"
        )
        state["messages"].append({"role": "assistant", "content": reply})
        logger.info("node=response auth_prompted=1")
        return state

    # Check if user just provided auth details and we should fetch order status
    # This handles the case where user provides email/order but intent wasn't detected as order_status
    if customer_authed and customer_email and order_number:
        # Check if user is asking about order status (current intent or in conversation history)
        # Look at recent user messages to see if they asked about order status
        recent_user_messages = [m.get("content", "").lower() for m in state["messages"][-5:] if m.get("role") == "user"]
        is_order_related = (
            intent == "order_status" or 
            "order" in user_text.lower() or 
            ("where" in user_text.lower() and "order" in user_text.lower()) or
            any("order" in msg or ("where" in msg and "order" in msg) for msg in recent_user_messages)
        )
        
        # If we have auth but no tool_result yet (or error), and user is asking about orders, fetch it now
        if is_order_related and (not tool_result or (isinstance(tool_result, dict) and tool_result.get("error"))):
            logger.info("node=response fetching_order_status_post_auth email=%s order=%s", customer_email, order_number)
            from tools import get_order_status
            tool_result = get_order_status(customer_email, order_number)
            state["tool_result"] = tool_result
            # Update intent to order_status if it wasn't already
            if intent != "order_status":
                state["intent"] = "order_status"

    # For order_status, compose a direct concise message including items and shipping
    if intent == "order_status" and isinstance(tool_result, dict) and not tool_result.get("error"):
        items = tool_result.get("items", [])
        items_str = ", ".join([f"{it.get('name')} (Qty: {it.get('quantity', 1)})" for it in items]) if items else ""
        parts = []
        parts.append(f"Order {tool_result.get('order_number')} is {tool_result.get('status')}")
        if items_str:
            parts.append(f"Items: {items_str}")
        if tool_result.get("carrier") and tool_result.get("tracking_id"):
            parts.append(f"Shipped via {tool_result.get('carrier')} (Tracking: {tool_result.get('tracking_id')})")
        if tool_result.get("eta"):
            parts.append(f"ETA: {tool_result.get('eta')}")
        if tool_result.get("shipping_address"):
            parts.append(f"Ship-to: {tool_result.get('shipping_address')}")
        reply = ". ".join(parts) + ".\n\nCan I help you with anything else?"
        state["messages"].append({"role": "assistant", "content": reply})
        logger.info("node=response composed_order_status reply_len=%d", len(reply))
        return state
    
    # Handle order_status errors
    if intent == "order_status" and isinstance(tool_result, dict) and tool_result.get("error"):
        error = tool_result.get("error")
        if error == "customer_not_found":
            reply = "I couldn't find an account with that email. Please check and try again. Can I help you with anything else?"
        elif error == "order_not_found":
            reply = f"I couldn't find order {order_number} for that email. Please verify your order number and try again. Can I help you with anything else?"
        else:
            reply = "I'm having trouble looking up your order right now. Please try again later. Can I help you with anything else?"
        state["messages"].append({"role": "assistant", "content": reply})
        logger.info("node=response order_status_error=%s", error)
        return state

    # Handle questions about recommended products
    if intent == "product_question":
        last_recs = state.get("last_recommendations", []) or []
        user_text_lower = user_text.lower()
        
        # Check if user is asking for NEW recommendations based on specific notes or product type
        # Patterns like "with lemon", "anything with rose", "single product not package"
        note_request_patterns = [
            "with ", "anything with", "something with", "perfume with",
            "that has", "which has", "containing"
        ]
        is_note_request = any(pattern in user_text_lower for pattern in note_request_patterns)
        
        # Check if user is changing product type (single product vs package/gift set)
        single_product_patterns = [
            "single product", "single perfume", "single bottle", "individual product",
            "not package", "no package", "not a package", "not collection", "no collection"
        ]
        package_patterns = [
            "package", "collection", "gift set", "set", "bundle"
        ]
        is_single_product_request = any(pattern in user_text_lower for pattern in single_product_patterns)
        is_package_request = any(pattern in user_text_lower for pattern in package_patterns) and not is_single_product_request
        
        # If user is asking for products with specific notes or changing product type, treat as new product_recommendation
        if is_note_request or is_single_product_request or is_package_request:
            # Re-route to product_recommendation flow to find new matches
            state["intent"] = "product_recommendation"
            logger.info("node=response re_routing_note_request_to_recommendation")
            # Let the recommendation flow handle this
            # Clear awaiting_choice to start fresh
            state["awaiting_choice"] = None
            # Continue to product_recommendation flow below
            intent = "product_recommendation"
        else:
            # Try to find product by name if mentioned - first check in last_recs
            product = None
            if last_recs:
                for rec in last_recs:
                    rec_name = rec.get("name", "").lower()
                    # Check if product name is mentioned in user text (match significant words, not common words)
                    rec_words = [w for w in rec_name.split() if len(w) > 2 and w not in ["the", "and", "for"]]
                    if rec_name and any(word in user_text_lower for word in rec_words):
                        product = rec
                        break
            
            # If product not found in last_recs, search entire product catalog
            if not product:
                # Extract potential product names from user query
                # Try to match against full product catalog
                product = get_product_by_name(user_text)
                if product:
                    logger.info("node=response found_product_in_catalog product_name=%s", product.get("name"))
                    # Add to last_recommendations so it can be referenced
                    state["last_recommendations"] = [product]
                    last_recs = [product]
            
            # Don't auto-select first product - let each handler decide what to do when product is None
            
            # Handle price questions
            if any(word in user_text_lower for word in ["price", "cost", "how much"]):
                # Check if multiple products in context and no specific product mentioned
                if len(last_recs) > 1 and not product:
                    # Ask LLM to determine if user is asking about total or specific product
                    import json
                    products_list = [p.get("name", "") for p in last_recs]
                    llm_prompt = f"""User has these products in context: {', '.join(products_list)}
User asked: "{user_text}"

Is the user asking about:
A) The total price of all products combined (phrases like "all of them", "all", "total", "everything", "combined")
B) The price of a specific product (mention which one if they specified)
C) Need clarification (they said "the price" without specifying which)

Reply with only 'TOTAL', the specific product name, or 'CLARIFY'."""
                    
                    llm_decision = _call_llm(llm_prompt).strip()
                    
                    if llm_decision == "TOTAL":
                        # Calculate and show total price
                        total_price = sum(p.get("price", 0) for p in last_recs)
                        product_names = [p.get("name", "") for p in last_recs]
                        reply = f"Your custom box includes {len(last_recs)} perfumes: {', '.join(product_names)}. Total price: ₹{total_price}. Want to see it or explore other options?"
                        
                        # Add BYOB page button for total price
                        buttons = [{"label": "View BYOB Page", "action": "open_url:https://blabliblulife.com/pages/build-your-own-box", "type": "success"}]
                        state["button_suggestions"] = buttons
                        state["messages"].append({"role": "assistant", "content": reply})
                        logger.info("node=response answered_total_price multiple_products=%d total=%d", len(last_recs), total_price)
                        return state
                    elif llm_decision == "CLARIFY":
                        # Ask for clarification with buttons
                        product_names = [p.get("name", "") for p in last_recs]
                        reply = f"Which perfume are you asking about? You have: {', '.join(product_names)}."
                        
                        # Add buttons for each product
                        buttons = []
                        for product in last_recs:
                            product_name = product.get("name", "")
                            if product_name:
                                buttons.append({"label": product_name, "action": f"select_product:{product.get('id')}", "type": "primary"})
                        
                        # Add "All of them" option
                        if len(last_recs) > 1:
                            buttons.append({"label": "All of them", "action": "select_product:all", "type": "secondary"})
                        
                        state["button_suggestions"] = buttons
                        state["messages"].append({"role": "assistant", "content": reply})
                        logger.info("node=response asking_which_product_for_price multiple_products=%d buttons=%d", len(last_recs), len(buttons))
                        return state
                    else:
                        # Try to find the mentioned product
                        for rec in last_recs:
                            if rec.get("name", "").lower() in llm_decision.lower():
                                product = rec
                                break
                
                # Fallback to first product if still None
                if not product:
                    if last_recs:
                        product = last_recs[0]
                    else:
                        # No product found at all
                        reply = "I'm not sure which perfume you're asking about. Could you tell me the product name, or would you like me to recommend something?"
                        state["messages"].append({"role": "assistant", "content": reply})
                        logger.info("node=response price_question_no_product=1")
                        return state
                
                price = product.get("price", 0)
                name = product.get("name", "this product")
                if price:
                    reply = f"The price of {name} is ₹{price}. Want to see it or explore other options?"
                else:
                    reply = f"I don't have the price information for {name} right now. Can I help you with anything else?"
                
                # Add action buttons
                buttons = []
                product_url = product.get("product_url", "")
                if product_url:
                    buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
                
                # Add other options button if more recommendations available
                current_index = state.get("recommendation_index", 0)
                remaining = len(last_recs) - current_index - 1
                if remaining > 0:
                    buttons.append({"label": "Other Options", "action": "show_other", "type": "primary"})
                
                # Add safer/bolder buttons
                current_price = product.get("price", 0)
                safer_exists = any(p.get("price", 0) < current_price for p in last_recs if p.get("id") != product.get("id"))
                if safer_exists:
                    buttons.append({"label": "Show Safer Option", "action": "show_safer", "type": "secondary"})
                
                bolder_exists = any(p.get("price", 0) > current_price for p in last_recs if p.get("id") != product.get("id"))
                if bolder_exists:
                    buttons.append({"label": "Show Bolder Option", "action": "show_bolder", "type": "secondary"})
                
                if buttons:
                    state["button_suggestions"] = buttons
                
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response answered_price_question=1 buttons=%d", len(buttons))
                return state
            
            # Handle occasion/when to wear questions
            if any(phrase in user_text_lower for phrase in ["when", "wear", "occasion", "use", "suitable"]):
                # Check if multiple products in context and no specific product mentioned
                if len(last_recs) > 1 and not product:
                    # Get recent conversation for better context
                    user_messages = [m["content"] for m in state["messages"] if m["role"] == "user"]
                    recent_context = " ".join(user_messages[-3:]) if len(user_messages) > 1 else user_text
                    
                    import json
                    products_list = [p.get("name", "") for p in last_recs]
                    llm_prompt = f"""User has these products in context: {', '.join(products_list)}
Recent conversation: {recent_context}
Current question: "{user_text}"

Did the user mention a specific product name in their question? Consider phrases like product names (e.g., "Selfmade", "Lights Off").

Reply with only the specific product name if clearly mentioned, or 'CLARIFY' if no specific product was mentioned."""
                    
                    llm_decision = _call_llm(llm_prompt).strip()
                    
                    if llm_decision == "CLARIFY":
                        # Ask for clarification with buttons
                        product_names = [p.get("name", "") for p in last_recs]
                        reply = f"Which perfume are you asking about? You have: {', '.join(product_names)}."
                        
                        # Add buttons for each product
                        buttons = []
                        for product in last_recs:
                            product_name = product.get("name", "")
                            if product_name:
                                buttons.append({"label": product_name, "action": f"select_product:{product.get('id')}", "type": "primary"})
                        
                        state["button_suggestions"] = buttons
                        state["messages"].append({"role": "assistant", "content": reply})
                        logger.info("node=response asking_which_product_for_occasion multiple_products=%d buttons=%d", len(last_recs), len(buttons))
                        return state
                    else:
                        # Try to find the mentioned product
                        for rec in last_recs:
                            if rec.get("name", "").lower() in llm_decision.lower():
                                product = rec
                                break
                        if not product:
                            product = last_recs[0]
                
                # Ensure we have a product selected
                if not product:
                    if last_recs:
                        product = last_recs[0]
                    else:
                        reply = "I'm not sure which perfume you're asking about. Could you tell me the product name?"
                        state["messages"].append({"role": "assistant", "content": reply})
                        logger.info("node=response occasion_question_no_product=1")
                        return state

                name = product.get("name", "this product")
                occasion_tags = product.get("occasion_tags", [])
                vibe_tags = product.get("vibe_tags", [])
                
                reply_parts = [f"{name} is great for"]
                
                if occasion_tags:
                    # Format occasions nicely
                    occasions = [occ.replace("_", " ") for occ in occasion_tags]
                    if len(occasions) == 1:
                        reply_parts.append(occasions[0])
                    elif len(occasions) == 2:
                        reply_parts.append(f"{occasions[0]} and {occasions[1]}")
                    else:
                        reply_parts.append(f"{', '.join(occasions[:-1])}, and {occasions[-1]}")
                else:
                    reply_parts.append("everyday wear")
                
                # Add vibe context
                if vibe_tags:
                    vibe_desc = ", ".join(vibe_tags[:3])
                    reply_parts.append(f"It has {vibe_desc} vibes.")
                
                reply = " ".join(reply_parts) + " Want to see it or explore other options?"
                
                # Add action buttons
                buttons = []
                product_url = product.get("product_url", "")
                if product_url:
                    buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
                
                # Add other options button if more recommendations available
                current_index = state.get("recommendation_index", 0)
                remaining = len(last_recs) - current_index - 1
                if remaining > 0:
                    buttons.append({"label": "Other Options", "action": "show_other", "type": "primary"})
                
                # Add safer/bolder buttons
                current_price = product.get("price", 0)
                current_vibes = product.get("vibe_tags", [])
                
                safer_candidates = [
                    p for p in last_recs 
                    if p.get("id") != product.get("id") and (
                        p.get("price", 0) < current_price or
                        any(v in p.get("vibe_tags", []) for v in ["fresh", "clean", "light", "breezy"])
                    )
                ]
                if safer_candidates:
                    buttons.append({"label": "Show Safer Option", "action": "show_safer", "type": "secondary"})
                
                bolder_candidates = [
                    p for p in last_recs 
                    if p.get("id") != product.get("id") and (
                        p.get("price", 0) > current_price or
                        any(v in p.get("vibe_tags", []) for v in ["bold", "intense", "strong", "oud", "spicy"])
                    )
                ]
                if bolder_candidates:
                    buttons.append({"label": "Show Bolder Option", "action": "show_bolder", "type": "secondary"})
                
                if buttons:
                    state["button_suggestions"] = buttons
                
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response answered_occasion_question=1 buttons=%d", len(buttons))
                return state
            
            # Handle questions about contents/what's included
            if any(phrase in user_text_lower for phrase in ["what do i get", "what's in", "what is in", "what comes", "contents", "includes"]):
                # Check if multiple products in context and no specific product mentioned
                if len(last_recs) > 1 and not product:
                    # Get recent conversation for better context
                    user_messages = [m["content"] for m in state["messages"] if m["role"] == "user"]
                    recent_context = " ".join(user_messages[-3:]) if len(user_messages) > 1 else user_text
                    
                    import json
                    products_list = [p.get("name", "") for p in last_recs]
                    llm_prompt = f"""User has these products in context: {', '.join(products_list)}
Recent conversation: {recent_context}
Current question: "{user_text}"

Did the user mention a specific product name in their question? Consider phrases like product names (e.g., "Selfmade", "Lights Off").

Reply with only the specific product name if clearly mentioned, or 'CLARIFY' if no specific product was mentioned."""
                    
                    llm_decision = _call_llm(llm_prompt).strip()
                    
                    if llm_decision == "CLARIFY":
                        # Ask for clarification with buttons
                        product_names = [p.get("name", "") for p in last_recs]
                        reply = f"Which perfume are you asking about? You have: {', '.join(product_names)}."
                        
                        # Add buttons for each product
                        buttons = []
                        for product in last_recs:
                            product_name = product.get("name", "")
                            if product_name:
                                buttons.append({"label": product_name, "action": f"select_product:{product.get('id')}", "type": "primary"})
                        
                        state["button_suggestions"] = buttons
                        state["messages"].append({"role": "assistant", "content": reply})
                        logger.info("node=response asking_which_product_for_contents multiple_products=%d buttons=%d", len(last_recs), len(buttons))
                        return state
                    else:
                        # Try to find the mentioned product
                        for rec in last_recs:
                            if rec.get("name", "").lower() in llm_decision.lower():
                                product = rec
                                break
                
                # Use the product found above, or first one
                if not product:
                    product = last_recs[0] if last_recs else {}
                name = product.get("name", "this product")
                desc = product.get("short_description", "")
                notes = product.get("notes", {})
                
                reply_parts = [f"{name}"]
                if desc:
                    reply_parts.append(f"includes: {desc}")
                
                # Extract notes if available
                all_notes = []
                if notes:
                    all_notes = (
                        notes.get("head", []) + 
                        notes.get("heart", []) + 
                        notes.get("base", [])
                    )
                
                if all_notes:
                    notes_str = ", ".join(all_notes[:5])  # Limit to first 5 notes
                    reply_parts.append(f"Key notes: {notes_str}")
                
                reply = ". ".join(reply_parts) + ". Want to see it or explore other options?"
                
                # Add action buttons
                buttons = []
                product_url = product.get("product_url", "")
                if product_url:
                    buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
                
                current_index = state.get("recommendation_index", 0)
                remaining = len(last_recs) - current_index - 1
                if remaining > 0:
                    buttons.append({"label": "Other Options", "action": "show_other", "type": "primary"})
                
                if buttons:
                    state["button_suggestions"] = buttons
                
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response answered_contents_question=1 buttons=%d", len(buttons))
                return state
            
            # Handle fragrance notes questions (including typo "nodes")
            if any(phrase in user_text_lower for phrase in ["what notes", "what nodes", "notes in", "nodes in", "fragrance notes", "which notes", "tell me about notes"]):
                # If no product found yet, try to find it from the query
                if not product:
                    # If we have last_recs but couldn't match, check if multiple products
                    if len(last_recs) > 1:
                        # Get recent conversation for better context
                        user_messages = [m["content"] for m in state["messages"] if m["role"] == "user"]
                        recent_context = " ".join(user_messages[-3:]) if len(user_messages) > 1 else user_text
                        
                        import json
                        products_list = [p.get("name", "") for p in last_recs]
                        llm_prompt = f"""User has these products in context: {', '.join(products_list)}
Recent conversation: {recent_context}
Current question: "{user_text}"

Did the user mention a specific product name in their question? Consider phrases like product names (e.g., "Selfmade", "Lights Off", "Love Drunk").

Reply with only the specific product name if clearly mentioned, or 'CLARIFY' if no specific product was mentioned."""
                        
                        llm_decision = _call_llm(llm_prompt).strip()
                        
                        if llm_decision == "CLARIFY":
                            # Ask for clarification with buttons
                            product_names = [p.get("name", "") for p in last_recs]
                            reply = f"Which perfume are you asking about? You have: {', '.join(product_names)}."
                            
                            # Add buttons for each product
                            buttons = []
                            for p_option in last_recs:
                                product_name = p_option.get("name", "")
                                if product_name:
                                    buttons.append({"label": product_name, "action": f"select_product:{p_option.get('id')}", "type": "primary"})
                            
                            state["button_suggestions"] = buttons
                            state["messages"].append({"role": "assistant", "content": reply})
                            logger.info("node=response asking_which_product_for_notes multiple_products=%d buttons=%d", len(last_recs), len(buttons))
                            return state
                        else:
                            # Try to find the mentioned product
                            for rec in last_recs:
                                if rec.get("name", "").lower() in llm_decision.lower():
                                    product = rec
                                    break
                    
                    # Still no product? Use first from last_recs or show error
                    if not product:
                        if last_recs:
                            product = last_recs[0]
                        else:
                            # No product in context, inform user
                            reply = "I'm not sure which perfume you're asking about. Could you tell me which one, or would you like me to recommend something?"
                            state["messages"].append({"role": "assistant", "content": reply})
                            logger.info("node=response notes_question_no_product_context=1")
                            return state
                
                # Now we have a product, extract and show notes
                name = product.get("name", "this product")
                notes = product.get("notes", {})
                
                if notes:
                    head_notes = notes.get("head", [])
                    heart_notes = notes.get("heart", [])
                    base_notes = notes.get("base", [])
                    
                    reply_parts = [f"{name} has"]
                    notes_desc = []
                    
                    if head_notes:
                        notes_desc.append(f"top notes of {', '.join(head_notes)}")
                    if heart_notes:
                        notes_desc.append(f"heart notes of {', '.join(heart_notes)}")
                    if base_notes:
                        notes_desc.append(f"base notes of {', '.join(base_notes)}")
                    
                    if notes_desc:
                        reply_parts.append("; ".join(notes_desc) + ".")
                    else:
                        reply_parts.append("a unique fragrance profile.")
                    
                    reply = " ".join(reply_parts) + " Want to see it or explore other options?"
                else:
                    # No notes data available
                    reply = f"{name} is a beautiful fragrance. Want to see it or explore other options?"
                
                # Add action buttons
                buttons = []
                product_url = product.get("product_url", "")
                if product_url:
                    buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
                
                current_index = state.get("recommendation_index", 0)
                remaining = len(last_recs) - current_index - 1
                if remaining > 0:
                    buttons.append({"label": "Other Options", "action": "show_other", "type": "primary"})
                
                if buttons:
                    state["button_suggestions"] = buttons
                
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response answered_notes_question=1 product=%s buttons=%d", name, len(buttons))
                return state
            
            # Handle other questions - use LLM but with structured product data
            import json
            products_context = json.dumps(last_recs[:2], indent=2)
            
            # Get conversation context (recent user messages for occasion/context)
            user_messages = [m["content"] for m in state["messages"] if m["role"] == "user"]
            conversation_context = " ".join(user_messages[-3:]) if len(user_messages) > 1 else ""
            
            # Get user preferences for context
            prefs = state.get("preferences") or {}
            occasion_context = ""
            if prefs.get("occasion_raw"):
                # Use raw occasion text (e.g., "wedding") for natural reference
                occasion_context = f"\nUser mentioned occasion: {prefs.get('occasion_raw')}"
            elif prefs.get("occasions"):
                occasions = prefs.get("occasions", [])
                occasion_context = f"\nUser mentioned occasion: {', '.join(occasions)}"
            
            prompt = f"""{ASSISTANT_STYLE_PROMPT}

The user is asking about these recommended products:
{products_context}

Conversation context: {conversation_context}
{occasion_context}

User question: {user_text}

IMPORTANT: 
- Extract information directly from the product data above. Do not make up prices or details.
- Price is in the "price" field (in rupees ₹)
- Description is in "short_description"
- Notes are in "notes" object with "head", "heart", "base" arrays
- Name is in "name" field
- Reference the conversation context (like wedding, occasion, etc.) in your answer to make it relevant
- DO NOT use markdown formatting, asterisks (**), or bold text - write in plain text only
- Keep response SHORT (2-3 sentences max) and conversational
- If user mentioned specific occasion earlier (wedding, date, party), reference it naturally

Answer their question naturally using ONLY the data provided above.
End with: "Want to see it or explore other options?"
"""
            answer = _call_llm(prompt)
            
            # Add action buttons
            buttons = []
            current_index = state.get("recommendation_index", 0)
            if current_index < len(last_recs):
                product = last_recs[current_index]
                product_url = product.get("product_url", "")
                if product_url:
                    buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
            
            remaining = len(last_recs) - current_index - 1
            if remaining > 0:
                buttons.append({"label": "Other Options", "action": "show_other", "type": "primary"})
            
            if buttons:
                state["button_suggestions"] = buttons
            
            state["messages"].append({"role": "assistant", "content": answer})
            logger.info("node=response answered_product_question=1 buttons=%d", len(buttons))
            return state
        
        # If we reach here, no product was found
        if not last_recs:
            # No product found - provide helpful response
            reply = "I couldn't find that product. Could you tell me what you're looking for? I can help you find the perfect perfume!"
            state["messages"].append({"role": "assistant", "content": reply})
            logger.info("node=response product_question_no_product_found=1")
            return state

    # Handle BYOB flow
    if intent == "build_own_box":
        if isinstance(tool_result, dict) and tool_result.get("byob_products"):
            byob_products = tool_result["byob_products"]
            
            # Check if user has already started selecting
            selections = state.get("byob_selections") or []
            max_items = state.get("byob_max_items", 3)
            
            # Check if we're awaiting a BYOB selection and user just selected a product
            # Use the saved value from before clearing
            awaiting_byob = awaiting_choice_before_clear == "byob_selection"
            logger.info("node=response byob_check awaiting=%s user_text=%s selections=%s", awaiting_byob, user_text, selections)
            
            if awaiting_byob:
                # Try to match user's message to a BYOB product name
                user_text_lower = user_text.lower().strip()
                selected_product_id = None
                
                # First check if message contains "select_byob:" action format
                if "select_byob:" in user_text:
                    try:
                        selected_product_id = user_text.split("select_byob:")[1].strip()
                        logger.info("node=response byob_found_action_format product_id=%s", selected_product_id)
                    except Exception as e:
                        logger.error("node=response byob_action_parse_error error=%s", e)
                        pass
                
                # If not found, try matching by product name
                if not selected_product_id:
                    logger.info("node=response byob_matching_by_name user_text_lower=%s products_count=%d", user_text_lower, len(byob_products))
                    for product in byob_products:
                        product_name = product.get("name", "").lower().strip()
                        product_id = product.get("id", "")
                        # More flexible matching: check if user text matches product name
                        # Remove common punctuation and extra spaces for better matching
                        product_name_clean = product_name.replace("-", " ").replace("  ", " ")
                        user_text_clean = user_text_lower.replace("-", " ").replace("  ", " ")
                        
                        # Check multiple matching strategies
                        if (product_name == user_text_lower or 
                            user_text_lower == product_name or
                            user_text_clean == product_name_clean or
                            user_text_lower in product_name or 
                            product_name in user_text_lower or
                            user_text_clean in product_name_clean or
                            product_name_clean in user_text_clean):
                            selected_product_id = product_id
                            logger.info("node=response byob_matched_product product_id=%s product_name=%s", product_id, product_name)
                            break
                
                if not selected_product_id:
                    logger.warning("node=response byob_no_match user_text=%s", user_text)
                
                # If we found a selection and it's not already selected
                if selected_product_id and selected_product_id not in selections:
                    selections.append(selected_product_id)
                    state["byob_selections"] = selections
                    logger.info("node=response byob_product_selected=1 product_id=%s total_selected=%d", selected_product_id, len(selections))
                    
                    # Check if we've reached the max
                    if len(selections) >= max_items:
                        # User has completed selection
                        validation = validate_byob_selection(selections, max_items)
                        if validation.get("valid"):
                            selected_products = validation["products"]
                            total_price = validation["total_price"]
                            
                            # Store selected products in last_recommendations for follow-up questions
                            state["last_recommendations"] = selected_products
                            
                            # Build summary
                            product_names = [p.get("name", "") for p in selected_products]
                            reply = f"Perfect! Your custom box includes: {', '.join(product_names)}. "
                            reply += f"Total: ₹{total_price}. "
                            reply += "Visit our BYOB page to complete your order. Can I help you with anything else?"
                            
                            # Add BYOB page link button
                            buttons = [{"label": "View BYOB Page", "action": "open_url:https://blabliblulife.com/pages/build-your-own-box", "type": "success"}]
                            state["button_suggestions"] = buttons
                        else:
                            reply = "There was an issue with your selections. Let's start over. Can I help you with anything else?"
                        
                        state["messages"].append({"role": "assistant", "content": reply})
                        # Clear selections and awaiting_choice after completion
                        state["byob_selections"] = []
                        state["awaiting_choice"] = None
                        logger.info("node=response byob_completed=1")
                        return state
                    else:
                        # Still need more selections - show updated list
                        remaining = max_items - len(selections)
                        reply = f"Great! You've selected {len(selections)}. Choose {remaining} more. "
                elif selected_product_id in selections:
                    # Product already selected
                    remaining = max_items - len(selections)
                    reply = f"You've already selected that one. You've selected {len(selections)}. Choose {remaining} more. "
                else:
                    # Couldn't match the selection - show products again
                    remaining = max_items - len(selections)
                    if len(selections) == 0:
                        reply = f"Great! You can choose {max_items} perfumes for your custom box. "
                    else:
                        reply = f"You've selected {len(selections)}. Choose {remaining} more. "
            else:
                # First time showing products
                remaining = max_items - len(selections)
                if len(selections) == 0:
                    reply = f"Great! You can choose {max_items} perfumes for your custom box. "
                else:
                    reply = f"You've selected {len(selections)}. Choose {remaining} more. "
            
            # Generate button suggestions for selection (exclude already selected)
            buttons = []
            for product in byob_products[:8]:  # Show top 8 options
                pid = product.get("id", "")
                name = product.get("name", "")
                if pid not in selections:
                    buttons.append({
                        "label": name,
                        "action": f"select_byob:{pid}",
                        "type": "primary"
                    })
            
            state["button_suggestions"] = buttons
            state["awaiting_choice"] = "byob_selection"
            
            reply += "Select your perfumes from the options below."
            state["messages"].append({"role": "assistant", "content": reply})
            logger.info("node=response byob_awaiting_selection remaining=%d selected=%d", remaining, len(selections))
            return state
        else:
            reply = "I couldn't load the BYOB products right now. Please try again. Can I help you with anything else?"
            state["messages"].append({"role": "assistant", "content": reply})
            logger.info("node=response byob_error=no_products")
            return state
    
    # Handle general questions (FAQ)
    if intent == "general_question":
        if isinstance(tool_result, dict) and tool_result.get("faq_answer"):
            reply = tool_result["faq_answer"] + " Can I help you with anything else?"
            state["messages"].append({"role": "assistant", "content": reply})
            logger.info("node=response general_question_answered=1")
            return state
        else:
            # No FAQ match - offer human handoff
            reply = "I'm not sure about that. Would you like to talk with a human agent who can help you better?"
            buttons = [
                {"label": "Yes, connect me", "action": "confirm_handoff", "type": "primary"},
                {"label": "No, thanks", "action": "cancel_handoff", "type": "secondary"},
            ]
            state["button_suggestions"] = buttons
            state["handoff_state"] = "awaiting_confirmation"
            state["messages"].append({"role": "assistant", "content": reply})
            logger.info("node=response general_question_no_match offering_handoff=1")
            return state
    
    # For recommendations: ask preferences one by one, remember from context
    if intent in ("product_recommendation", "find_perfume", "gift_recommendation"):
        # Check if we have sufficient preferences before recommending
        prefs = state.get("preferences") or {}
        has_gender = prefs.get("gender") is not None
        has_scent_type = isinstance(prefs.get("scent_types"), list) and len(prefs.get("scent_types", [])) > 0
        has_budget = prefs.get("budget") is not None
        
        # If intent is gift_recommendation, mark as gift
        is_gift = prefs.get("is_gift", False) or intent == "gift_recommendation"
        if intent == "gift_recommendation" and not prefs.get("is_gift"):
            prefs["is_gift"] = True
            state["preferences"] = prefs
        
        # Initialize questions_asked if not present
        if "questions_asked" not in state:
            state["questions_asked"] = []
        
        questions_asked = state.get("questions_asked", [])
        
        # If gender was auto-detected (e.g., from "girlfriend", "boyfriend"), mark it as asked
        if has_gender and "gender" not in questions_asked:
            questions_asked.append("gender")
            state["questions_asked"] = questions_asked
        
        has_occasion = isinstance(prefs.get("occasions"), list) and len(prefs.get("occasions", [])) > 0
        
        # For gift queries, follow the flow from bot_suggestion_flow.txt:
        # Ask gender -> vibe (personality) -> budget (3 questions before recommending)
        # For regular queries, follow the flow from bot_suggestion_flow.txt:
        # Ask gender -> vibe -> occasion -> budget (all 4 questions before recommending)
        if is_gift:
            # Gifts need: gender, vibe (personality), budget
            # Check if these specific questions have been asked
            gift_required_questions = {"gender", "vibe", "budget"}
            has_sufficient_prefs = gift_required_questions.issubset(set(questions_asked))
        else:
            # Only recommend after ALL 4 questions have been asked
            # Questions: gender, vibe, occasion, budget
            required_questions = {"gender", "vibe", "occasion", "budget"}
            has_sufficient_prefs = required_questions.issubset(set(questions_asked))
        
        # Only recommend if we have sufficient preferences AND recommendations exist
        if has_sufficient_prefs:
            # Get recommendations from tool_result (fresh) or from state (stored from previous turn)
            recs = None
            if isinstance(tool_result, dict) and tool_result.get("recommendations"):
                recs = tool_result["recommendations"]
            elif state.get("recommendations"):
                recs = state.get("recommendations")
            
            if recs:
                
                # Check what the user is asking for
                user_text_lower = user_text.lower()
                asking_for_safer = any(phrase in user_text_lower for phrase in ["safer option", "safer", "lighter", "softer"])
                asking_for_bolder = any(phrase in user_text_lower for phrase in ["bolder option", "bolder", "stronger", "more intense"])
                asking_for_other = any(phrase in user_text_lower for phrase in [
                    "other option", "another option", "other options", "anything else", "something else",
                    "show me another", "different one", "what else", "more options"
                ])
                
                # Get current recommendation index (default to 0 if not set)
                current_index = state.get("recommendation_index", 0)
                current_product = recs[current_index] if current_index < len(recs) else None
                
                # Handle safer/bolder/other options
                # Get user's gender preference to filter recommendations
                user_gender = prefs.get("gender")
                
                if asking_for_safer and current_product:
                    # Find the safest option (lowest price or lightest vibe) matching user's gender
                    current_price = current_product.get("price", 0)
                    safer_recs = [
                        (idx, p) for idx, p in enumerate(recs)
                        if idx != current_index and (
                            # Match gender if specified, or accept unisex
                            (not user_gender or p.get("gender_profile") == user_gender or p.get("gender_profile") == "unisex")
                        ) and (
                            p.get("price", 0) < current_price or
                            any(v in p.get("vibe_tags", []) for v in ["fresh", "clean", "light", "breezy"])
                        )
                    ]
                    if safer_recs:
                        # Sort by price (ascending) and pick the first
                        safer_recs.sort(key=lambda x: x[1].get("price", 0))
                        current_index = safer_recs[0][0]
                elif asking_for_bolder and current_product:
                    # Find the boldest option (highest price or strongest vibe) matching user's gender
                    current_price = current_product.get("price", 0)
                    bolder_recs = [
                        (idx, p) for idx, p in enumerate(recs)
                        if idx != current_index and (
                            # Match gender if specified, or accept unisex
                            (not user_gender or p.get("gender_profile") == user_gender or p.get("gender_profile") == "unisex")
                        ) and (
                            p.get("price", 0) > current_price or
                            any(v in p.get("vibe_tags", []) for v in ["bold", "intense", "strong", "oud", "spicy"])
                        )
                    ]
                    if bolder_recs:
                        # Sort by price (descending) and pick the first
                        bolder_recs.sort(key=lambda x: x[1].get("price", 0), reverse=True)
                        current_index = bolder_recs[0][0]
                elif asking_for_other:
                    # Just show the next product in the list
                    current_index += 1
                
                # Check if we have more recommendations to show
                if current_index < len(recs):
                    top = recs[current_index]
                    # Store all recommendations internally for navigation, but only show the current one
                    state["recommendations"] = recs  # Keep all recommendations for "Other Options" navigation
                    state["last_recommendations"] = [top]  # Store only the currently shown product
                    state["recommendation_index"] = current_index  # Update index
                    
                    name = top.get("name", "")
                    positioning = top.get("positioning_line", "")
                    desc = top.get("short_description", "")
                    price = top.get("price", 0)
                    
                    # Build a natural response with context awareness
                    reply_parts = []
                    
                    # Add context prefix based on what user selected
                    if asking_for_other:
                        reply_parts.append(f"Here's another option: {name}.")
                    elif asking_for_bolder:
                        reply_parts.append(f"Here's a bolder option: {name}.")
                    elif asking_for_safer:
                        reply_parts.append(f"Here's a safer option: {name}.")
                    else:
                        reply_parts.append(f"I'd suggest {name}.")
                    
                    # Keep response short - combine positioning and description
                    if positioning:
                        reply_parts.append(positioning)
                    elif desc:
                        reply_parts.append(desc)
                    
                    if price:
                        reply_parts.append(f"Priced at ₹{price}.")
                    
                    # Add button suggestions for safer/bolder options and other actions
                    buttons = []
                    remaining = len(recs) - current_index - 1
                    
                    # Add safer/bolder option buttons based on product characteristics
                    # Safer = lower price or lighter scent, Bolder = higher price or stronger scent
                    current_price = top.get("price", 0)
                    current_vibes = top.get("vibe_tags", [])
                    
                    # Find safer option (lighter/cheaper) matching user's gender
                    safer_candidates = [
                        (idx, p) for idx, p in enumerate(recs) 
                        if idx != current_index and (
                            # Match gender if specified, or accept unisex
                            (not user_gender or p.get("gender_profile") == user_gender or p.get("gender_profile") == "unisex")
                        ) and (
                            p.get("price", 0) < current_price or
                            any(v in p.get("vibe_tags", []) for v in ["fresh", "clean", "light", "breezy"])
                        )
                    ]
                    if safer_candidates:
                        buttons.append({"label": "Show Safer Option", "action": "show_safer", "type": "secondary"})
                    
                    # Find bolder option (stronger/pricier) matching user's gender
                    bolder_candidates = [
                        (idx, p) for idx, p in enumerate(recs) 
                        if idx != current_index and (
                            # Match gender if specified, or accept unisex
                            (not user_gender or p.get("gender_profile") == user_gender or p.get("gender_profile") == "unisex")
                        ) and (
                            p.get("price", 0) > current_price or
                            any(v in p.get("vibe_tags", []) for v in ["bold", "intense", "strong", "oud", "spicy"])
                        )
                    ]
                    if bolder_candidates:
                        buttons.append({"label": "Show Bolder Option", "action": "show_bolder", "type": "secondary"})
                    
                    # Add "other options" button if more products available
                    if remaining > 0:
                        buttons.append({"label": "Other Options", "action": "show_other", "type": "primary"})
                        reply_parts.append("Want to know more or see other options?")
                    else:
                        reply_parts.append("Want to know more?")
                    
                    # Add "View Product" button
                    product_url = top.get("product_url", "")
                    if product_url:
                        buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
                    
                    if buttons:
                        state["button_suggestions"] = buttons
                    
                    reply = " ".join(reply_parts)
                    
                    state["messages"].append({"role": "assistant", "content": reply})
                    logger.info("node=response used_recommendations=1 showing_product_index=%d remaining=%d buttons=%d", 
                               current_index, remaining, len(buttons))
                    return state
                else:
                    # No more recommendations
                    total_shown = len(recs)
                    if total_shown == 1:
                        reply = "That's the only option matching your preferences. Would you like to adjust your requirements?"
                    else:
                        reply = f"Those are the {total_shown} options matching your preferences. Would you like to adjust your requirements to see more?"
                    state["messages"].append({"role": "assistant", "content": reply})
                    logger.info("node=response exhausted_recommendations total_shown=%d", total_shown)
                    return state
            else:
                # Sufficient preferences but NO recommendations found (e.g. strict budget filter)
                # Provide helpful suggestions based on what might be blocking
                reply_parts = ["I couldn't find any perfumes matching your exact criteria."]
                
                # Build suggestions based on preferences
                suggestions = []
                buttons = []
                
                # Check budget constraint
                budget = prefs.get("budget")
                if budget and budget.get("operator") == "under":
                    amount = budget.get("amount", 0)
                    suggestions.append(f"try a higher budget (above ₹{amount})")
                    buttons.append({"label": f"Above ₹{amount}", "action": f"select_budget:above_{amount}", "type": "primary"})
                elif budget and budget.get("operator") == "over":
                    amount = budget.get("amount", 0)
                    suggestions.append(f"try a lower budget (under ₹{amount})")
                    buttons.append({"label": f"Under ₹{amount}", "action": f"select_budget:under_{amount}", "type": "primary"})
                
                # Check occasion constraint
                occasions = prefs.get("occasions", [])
                if occasions and len(occasions) == 1:
                    # Suggest alternative occasions
                    current_occasion = occasions[0]
                    occasion_alternatives = {
                        "formal": ("casual", "Daily/Casual"),
                        "casual": ("formal", "Office/Formal"),
                        "date_night": ("party", "Party/Night Out"),
                        "party": ("casual", "Daily/Casual"),
                    }
                    if current_occasion in occasion_alternatives:
                        alt_key, alt_label = occasion_alternatives[current_occasion]
                        suggestions.append(f"try {alt_label.lower()}")
                        buttons.append({"label": alt_label, "action": f"select_occasion:{alt_key}", "type": "primary"})
                
                # Check scent type constraint
                scent_types = prefs.get("scent_types", [])
                if scent_types and len(scent_types) == 1:
                    # Suggest alternative scent types
                    current_scent = scent_types[0]
                    scent_alternatives = {
                        "oud": ("woody", "Woody/Earthy"),
                        "spicy": ("fresh", "Fresh/Breezy"),
                        "fresh": ("sweet", "Sweet/Gourmand"),
                        "sweet": ("floral", "Floral/Soft"),
                        "floral": ("fresh", "Fresh/Breezy"),
                        "woody": ("spicy", "Spicy/Bold"),
                    }
                    if current_scent in scent_alternatives:
                        alt_key, alt_label = scent_alternatives[current_scent]
                        suggestions.append(f"try {alt_label.lower()}")
                        buttons.append({"label": alt_label, "action": f"select_vibe:{alt_key}", "type": "primary"})
                
                # Build final reply
                if suggestions:
                    reply_parts.append(f"Would you like to {' or '.join(suggestions)}?")
                else:
                    reply_parts.append("Would you like to adjust your preferences?")
                    # Generic buttons if no specific suggestions
                    buttons = [
                        {"label": "Change Budget", "action": "change_budget", "type": "primary"},
                        {"label": "Change Occasion", "action": "change_occasion", "type": "primary"},
                        {"label": "Change Vibe", "action": "change_vibe", "type": "primary"},
                    ]
                
                reply = " ".join(reply_parts)
                
                if buttons:
                    state["button_suggestions"] = buttons
                
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response no_recommendations_found sufficient_prefs=1 suggestions=%d", len(suggestions))
                return state
        
        # Ask for missing preferences ONE AT A TIME with button suggestions
        # Priority: gender > vibe/scent_type > occasion > budget
        buttons = []
        
        # Use saved awaiting_choice for preference detection
        current_awaiting = awaiting_choice_before_clear
        
        if is_gift:
            # Gift flow: ask gender -> vibe (personality) -> budget
            if not has_gender:
                reply = "Who is this gift for?"
                buttons = [
                    {"label": "For Him", "action": "select_gender:for_him", "type": "primary"},
                    {"label": "For Her", "action": "select_gender:for_her", "type": "primary"},
                    {"label": "Unisex", "action": "select_gender:unisex", "type": "secondary"},
                ]
                state["awaiting_choice"] = "gender"
                if "gender" not in questions_asked:
                    questions_asked.append("gender")
                    state["questions_asked"] = questions_asked
            elif "vibe" not in questions_asked:
                # Have gender, ask about recipient's personality (maps to vibe/scent)
                # Use appropriate pronouns based on gift recipient gender
                gender = prefs.get("gender")
                if gender == "for_him":
                    reply = "What's his personality like?"
                elif gender == "for_her":
                    reply = "What's her personality like?"
                else:
                    reply = "What's their personality like?"
                
                buttons = [
                    {"label": "Classy & Elegant", "action": "select_vibe:floral", "type": "primary"},
                    {"label": "Playful & Fun", "action": "select_vibe:fresh", "type": "primary"},
                    {"label": "Sweet & Romantic", "action": "select_vibe:sweet", "type": "primary"},
                    {"label": "Intense & Bold", "action": "select_vibe:spicy", "type": "primary"},
                ]
                state["awaiting_choice"] = "vibe"
                questions_asked.append("vibe")
                state["questions_asked"] = questions_asked
            elif "budget" not in questions_asked:
                # Have gender and vibe, ask for budget
                reply = "What's your budget range?"
                buttons = [
                    {"label": "Under ₹700", "action": "select_budget:under_700", "type": "primary"},
                    {"label": "₹700-₹900", "action": "select_budget:700_900", "type": "primary"},
                    {"label": "No Limit", "action": "select_budget:no_limit", "type": "secondary"},
                ]
                state["awaiting_choice"] = "budget"
                questions_asked.append("budget")
                state["questions_asked"] = questions_asked
            else:
                # All gift questions asked, make recommendation
                reply = "Let me find the perfect gift set for you."
        else:
            # Regular product recommendation flow - ask ONE question at a time
            if not has_gender:
                reply = "Who is this for?"
                buttons = [
                    {"label": "For Him", "action": "select_gender:for_him", "type": "primary"},
                    {"label": "For Her", "action": "select_gender:for_her", "type": "primary"},
                    {"label": "Unisex", "action": "select_gender:unisex", "type": "secondary"},
                ]
                state["awaiting_choice"] = "gender"
                if "gender" not in questions_asked:
                    questions_asked.append("gender")
                    state["questions_asked"] = questions_asked
            elif not has_scent_type:
                # Have gender, ask for vibe/scent type next
                reply = "What kind of vibe are you looking for?"
                buttons = [
                    {"label": "Fresh & Breezy", "action": "select_vibe:fresh", "type": "primary"},
                    {"label": "Sweet & Gourmand", "action": "select_vibe:sweet", "type": "primary"},
                    {"label": "Spicy & Bold", "action": "select_vibe:spicy", "type": "primary"},
                    {"label": "Oud & Intense", "action": "select_vibe:oud", "type": "primary"},
                    {"label": "Soft & Cozy", "action": "select_vibe:floral", "type": "secondary"},
                    {"label": "Woody & Earthy", "action": "select_vibe:woody", "type": "secondary"},
                ]
                state["awaiting_choice"] = "vibe"
                if "vibe" not in questions_asked:
                    questions_asked.append("vibe")
                    state["questions_asked"] = questions_asked
            elif "occasion" not in questions_asked:
                # Have gender and scent, ask for occasion
                reply = "When will you wear it?"
                buttons = [
                    {"label": "Daily", "action": "select_occasion:casual", "type": "primary"},
                    {"label": "Office", "action": "select_occasion:formal", "type": "primary"},
                    {"label": "Date Night", "action": "select_occasion:date_night", "type": "primary"},
                    {"label": "Party", "action": "select_occasion:party", "type": "primary"},
                    {"label": "Mixed", "action": "select_occasion:mixed", "type": "secondary"},
                ]
                state["awaiting_choice"] = "occasion"
                questions_asked.append("occasion")
                state["questions_asked"] = questions_asked
            elif "budget" not in questions_asked:
                # Have all other prefs, ask for budget
                reply = "What's your budget range?"
                buttons = [
                    {"label": "Under ₹700", "action": "select_budget:under_700", "type": "primary"},
                    {"label": "₹700-₹900", "action": "select_budget:700_900", "type": "primary"},
                    {"label": "No Limit", "action": "select_budget:no_limit", "type": "secondary"},
                ]
                state["awaiting_choice"] = "budget"
                questions_asked.append("budget")
                state["questions_asked"] = questions_asked
            else:
                # All questions have been asked, make recommendation
                reply = "Let me find the perfect perfume for you."
        
        if buttons:
            state["button_suggestions"] = buttons
        
        state["messages"].append({"role": "assistant", "content": reply})
        logger.info("node=response ask_clarifying_for_reco=1 missing_gender=%s missing_scent=%s missing_budget=%s is_gift=%s buttons=%d", 
                   not has_gender, not has_scent_type, not has_budget, is_gift, len(buttons))
        return state

    # If product_info but no exact match, check if user is asking about previously recommended products
    if intent == "product_info":
        last_recs = state.get("last_recommendations", [])
        if last_recs:
            # User might be asking about a previously recommended product
            # Check if the question matches product_question patterns
            user_text_lower = user_text.lower()
            if any(phrase in user_text_lower for phrase in ["what", "price", "cost", "contents", "includes", "what's in", "what do i get"]):
                # Treat as product_question instead
                product = last_recs[0] if last_recs else {}
                price = product.get("price", 0)
                name = product.get("name", "this product")
                desc = product.get("short_description", "")
                notes = product.get("notes", {})
                
                if "price" in user_text_lower or "cost" in user_text_lower:
                    if price:
                        reply = f"The price of {name} is ₹{price}. Can I help you with anything else?"
                    else:
                        reply = f"I don't have the price information for {name} right now. Can I help you with anything else?"
                elif any(phrase in user_text_lower for phrase in ["what do i get", "what's in", "what is in", "contents", "includes"]):
                    reply_parts = [f"{name}"]
                    if desc:
                        reply_parts.append(f"includes: {desc}")
                    all_notes = []
                    if notes:
                        all_notes = (
                            notes.get("head", []) + 
                            notes.get("heart", []) + 
                            notes.get("base", [])
                        )
                    if all_notes:
                        notes_str = ", ".join(all_notes[:5])
                        reply_parts.append(f"Key notes: {notes_str}")
                    reply = ". ".join(reply_parts) + ". Can I help you with anything else?"
                else:
                    # Generic product info question
                    reply_parts = [f"{name}"]
                    if desc:
                        reply_parts.append(desc)
                    if price:
                        reply_parts.append(f"Priced at ₹{price}.")
                    reply = ". ".join(reply_parts) + " Can I help you with anything else?"
                
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response answered_product_info_with_context=1")
                return state
        
        # No last recommendations - proceed with normal product_info flow
        if isinstance(tool_result, dict) and tool_result.get("products") and not tool_result.get("product"):
            reply = (
                "We have a few options. To narrow it down, what gender are you looking for (men's, women's, or unisex), "
                "what scent type do you prefer, and what's your budget? "
                "I can recommend one that fits best. Can I help you with anything else?"
            )
            state["messages"].append({"role": "assistant", "content": reply})
            logger.info("node=response ask_clarifying_for_product_info=1")
            return state

    # Handle fallback intent - offer human handoff if bot doesn't understand
    if intent == "fallback":
        reply = "I'm not sure I understand. Would you like to talk with a human agent who can help you better?"
        buttons = [
            {"label": "Yes, connect me", "action": "confirm_handoff", "type": "primary"},
            {"label": "No, thanks", "action": "cancel_handoff", "type": "secondary"},
        ]
        state["button_suggestions"] = buttons
        state["handoff_state"] = "awaiting_confirmation"
        state["messages"].append({"role": "assistant", "content": reply})
        logger.info("node=response fallback_intent offering_handoff=1")
        return state

    tool_context = ""
    if tool_result:
        tool_context = f"Tool result: {tool_result}"

    final_prompt = f"""{ASSISTANT_STYLE_PROMPT}

User said: {user_text}
Detected intent: {intent}
{tool_context}

Write a friendly, human, short answer. Do not use Markdown or lists. Keep it conversational.
End with: "Can I help you with anything else?"
"""
    answer = _call_llm(final_prompt)
    logger.info("node=response llm_called=1 answer_len=%d", len(answer) if isinstance(answer, str) else -1)

    state["messages"].append({"role": "assistant", "content": answer})
    return state


graph = StateGraph(ChatState)
graph.set_entry_point("intent_router")
graph.add_node("intent_router", intent_router)
graph.add_node("auth_guard", auth_guard)
graph.add_node("collect_preferences", collect_preferences)
graph.add_node("tool_exec", tool_exec)
graph.add_node("response_node", response_node)

graph.add_edge("intent_router", "auth_guard")
graph.add_edge("auth_guard", "collect_preferences")
graph.add_edge("collect_preferences", "tool_exec")
graph.add_edge("tool_exec", "response_node")
graph.add_edge("response_node", END)

compiled_app = graph.compile()


import os
from typing import TypedDict, List, Optional, Dict, Any
import re
import logging
import time

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from openai import OpenAI

from prompts import INTENT_PROMPT, ASSISTANT_STYLE_PROMPT, PREFERENCE_EXTRACTION_PROMPT
from tools import (
    get_product_by_name,
    list_products,
    recommend_products,
    get_order_status,
    get_return_policy,
    get_byob_products,
    validate_byob_selection,
    get_faq_answer,
)

try:
    from recommender import extract_preferences_and_recommend
except Exception:
    extract_preferences_and_recommend = None

logger = logging.getLogger("blabli.graph")
_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger.setLevel(_level)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
    logger.addHandler(_h)
logger.propagate = False


def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Put it in .env at project root.")
    return OpenAI(api_key=api_key)


class ChatState(TypedDict, total=False):
    messages: List[dict]
    intent: Optional[str]
    customer_authed: bool
    customer_email: Optional[str]
    order_number: Optional[str]
    tool_result: Optional[dict]
    missing_field: Optional[str]
    preferences: Optional[Dict[str, Any]]
    recommendations: Optional[List[Dict[str, Any]]]
    last_recommendations: Optional[List[Dict[str, Any]]]  # Track last recommended products for follow-up questions
    recommendation_index: int  # Track which recommendation we've shown (0-indexed)
    byob_selections: Optional[List[str]]  # Track BYOB product selections (IDs)
    byob_max_items: int  # Max items allowed in BYOB (default 3)
    button_suggestions: Optional[List[Dict[str, str]]]  # Button suggestions to show to user
    awaiting_choice: Optional[str]  # Track what choice we're waiting for (vibe, occasion, gender, etc.)
    questions_asked: Optional[List[str]]  # Track which questions have been asked (gender, vibe, occasion, budget)
    handoff_state: Optional[str]  # Track human handoff flow: None, "awaiting_confirmation", "collecting_email", "collecting_phone", "completed"
    handoff_email: Optional[str]  # Email collected for handoff
    handoff_phone: Optional[str]  # Phone collected for handoff
    handoff_completed: bool  # Flag to stop bot after handoff


def _call_llm(prompt: str) -> str:
    """
    Wrapper that works with both the modern OpenAI Responses API (>=1.0)
    and the legacy Chat Completions API (<1.0).
    """
    client = get_client()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if hasattr(client, "responses"):
        resp = client.responses.create(model=model, input=prompt)
        return resp.output[0].content[0].text.strip()

    # fallback for older openai package versions
    if not hasattr(client, "chat") or not hasattr(client.chat, "completions"):
        raise RuntimeError("OpenAI client does not support responses or chat completions APIs.")

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    message = resp.choices[0].message
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    return content.strip()


def _call_intent_llm(prompt: str) -> str:
    # Prompt is now constructed by the caller (intent_router) to include context
    return _call_llm(prompt)


def intent_router(state: ChatState):
    logger.info("node=intent_router entering")
    last_user_msg = ""
    last_assistant_msg = ""
    
    # Find last user message and last assistant message
    for m in reversed(state["messages"]):
        if m["role"] == "user" and not last_user_msg:
            last_user_msg = m["content"]
        elif m["role"] == "assistant" and not last_assistant_msg:
            last_assistant_msg = m["content"]
        
        if last_user_msg and last_assistant_msg:
            break
    
    # Direct intent mapping for button actions (bypassing LLM for efficiency)
    user_msg_lower = last_user_msg.lower()
    
    # Handle handoff button actions
    if "confirm_handoff" in last_user_msg or user_msg_lower in ["yes", "yeah", "yep", "sure", "ok", "okay", "connect me"]:
        handoff_state = state.get("handoff_state")
        if handoff_state == "awaiting_confirmation":
            state["intent"] = "human_handoff"
            logger.info("node=intent_router handoff_confirmed=1")
            return state
    
    # Special handling: If user is answering a direct choice question with a button-like response
    # Only continue intent if they're literally answering the question we asked
    awaiting_choice = state.get("awaiting_choice")
    if awaiting_choice:
        last_intent = state.get("intent")
        # Check if this is a button-style answer to the question we asked
        is_button_answer = False
        
        if awaiting_choice == "byob_selection":
            # If we're awaiting BYOB selection, preserve build_own_box intent
            # User might be clicking a product button or typing a product name
            if last_intent == "build_own_box":
                logger.info("node=intent_router continuing_intent=build_own_box awaiting=byob_selection")
                state["intent"] = "build_own_box"
                return state
        elif awaiting_choice == "gender" and any(word in user_msg_lower for word in ["for him", "for her", "unisex", "men's", "women's"]):
            is_button_answer = True
        elif awaiting_choice == "vibe" and any(word in user_msg_lower for word in ["fresh", "sweet", "spicy", "oud", "floral", "woody", "breezy", "gourmand", "cozy", "earthy"]):
            is_button_answer = True
        elif awaiting_choice == "occasion" and any(word in user_msg_lower for word in ["daily", "office", "date night", "party", "mixed", "casual", "formal"]):
            is_button_answer = True
        elif awaiting_choice == "budget" and any(word in user_msg_lower for word in ["under", "700", "900", "no limit", "any", "₹"]):
            is_button_answer = True
        
        if is_button_answer and last_intent in ("find_perfume", "product_recommendation", "gift_recommendation"):
            logger.info("node=intent_router continuing_intent=%s awaiting=%s detected_button_answer=1", last_intent, awaiting_choice)
            state["intent"] = last_intent
            return state
    
    direct_intent_map = {
        "find my perfume": "find_perfume",
        "gift for someone": "gift_recommendation",
        "build my own box": "build_own_box",
        "track my order": "order_status",
        "looking for perfume": "find_perfume",
        "want a gift": "gift_recommendation",
        "create my box": "build_own_box",
        "track order": "order_status",
        "talk to human": "human_handoff",
        "speak with human": "human_handoff",
        "talk to agent": "human_handoff",
        "speak with agent": "human_handoff",
        "talk to representative": "human_handoff",
        "speak with representative": "human_handoff",
        "talk to support": "human_handoff",
        "speak with support": "human_handoff",
        "human agent": "human_handoff",
        "connect me to human": "human_handoff",
    }
    
    for phrase, mapped_intent in direct_intent_map.items():
        if phrase in user_msg_lower:
            state["intent"] = mapped_intent
            logger.info("node=intent_router direct_map_intent=%s", mapped_intent)
            return state
    
    # Check if this is a product question about the last recommended product
    has_last_recs = bool(state.get("last_recommendations"))
    if has_last_recs:
        # Common product question patterns
        product_question_patterns = [
            "what is the price", "what is its price", "how much", "what's the price",
            "what do i get", "what's in it", "what comes with", "contents",
            "when can i wear", "when to wear", "when should i wear", "suitable for",
            "what occasion", "what notes", "tell me more", "more about",
        ]
        if any(pattern in user_msg_lower for pattern in product_question_patterns):
            state["intent"] = "product_question"
            logger.info("node=intent_router direct_map_product_question=1")
            return state
    
    # Check if user is asking about previously recommended products
    # This helps improve intent detection for product_question
    
    # Construct prompt with context
    prompt = INTENT_PROMPT
    if last_assistant_msg:
        prompt += f"\nAssistant asked: {last_assistant_msg}"
    
    if has_last_recs:
        # Add context to help intent detection
        prompt += f"\nContext: Products were just recommended."
        
    prompt += f"\nUser: {last_user_msg}"
    
    intent = _call_intent_llm(prompt)
    
    state["intent"] = intent
    logger.info("node=intent_router detected_intent=%s has_last_recs=%s", intent, has_last_recs)
    return state


def auth_guard(state: ChatState):
    logger.info(
        "node=auth_guard intent=%s email_present=%s order_present=%s",
        state.get("intent"),
        bool(state.get("customer_email")),
        bool(state.get("order_number")),
    )
    # If we're in an auth-requiring flow, try to extract missing fields from latest user message
    requires_auth = state["intent"] in ("order_status", "change_address") or state.get("missing_field") == "auth"

    if requires_auth:
        last_user_msg = ""
        for m in reversed(state["messages"]):
            if m["role"] == "user":
                last_user_msg = m["content"]
                break

        # Fill email if missing
        if not state.get("customer_email") and last_user_msg:
            email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", last_user_msg)
            if email_match:
                state["customer_email"] = email_match.group(0)
                logger.info("node=auth_guard extracted_email=1")

        # Fill order number if missing (e.g., BLB-1001)
        if not state.get("order_number") and last_user_msg:
            order_match = re.search(r"\bBLB-\d{3,}\b", last_user_msg, flags=re.IGNORECASE)
            if order_match:
                state["order_number"] = order_match.group(0).upper()
                logger.info("node=auth_guard extracted_order_number=1")

        if not state.get("customer_email") or not state.get("order_number"):
            state["missing_field"] = "auth"
            state["customer_authed"] = False
            logger.info("node=auth_guard auth_status=missing")
        else:
            state["missing_field"] = None
            state["customer_authed"] = True
            logger.info("node=auth_guard auth_status=ok")
    else:
        state["missing_field"] = None
    return state


def collect_preferences(state: ChatState):
    """
    For product-related intents, normalize or initialize the user's preference profile
    before we hit tools / LLM. This makes the bot contextual across turns.
    """
    intent = state.get("intent")
    if intent not in ("product_info", "product_recommendation", "find_perfume", "gift_recommendation"):
        return state

    # Get all user messages from conversation history to extract preferences from context
    user_messages = [m["content"] for m in state["messages"] if m["role"] == "user"]
    last_user_msg = user_messages[-1] if user_messages else ""
    last_user_msg_lower = last_user_msg.lower()
    # Combine recent user messages for better context (last 3 messages)
    context_text = " ".join(user_messages[-3:]) if len(user_messages) > 0 else ""

    # ensure preferences dict exists (adapted for perfumes) - preserve existing preferences
    prefs = state.get("preferences") or {
        "gender": None,
        "scent_types": [],
        "budget": None,
        "occasions": [],
        "vibes": [],
        "specific_notes": [],
        "is_gift": False,
        "occasion_raw": None,  # Store raw user input for occasion (e.g., "wedding", "date night")
    }
    
    # Direct preference detection from button-like responses (faster than LLM)
    # This handles cases where user types instead of clicking buttons
    awaiting_choice = state.get("awaiting_choice")
    
    if awaiting_choice == "gender" and not prefs.get("gender"):
        # User was asked about gender, detect their answer
        # Check for specific gender terms first (more specific patterns win)
        if any(word in last_user_msg_lower for word in ["men's", "mens", "for him", "male perfume", "man's", "guy's"]):
            prefs["gender"] = "for_him"
            logger.info("node=collect_preferences direct_detect_gender=for_him")
        elif any(word in last_user_msg_lower for word in ["women's", "womens", "for her", "female perfume", "woman's", "lady's"]):
            prefs["gender"] = "for_her"
            logger.info("node=collect_preferences direct_detect_gender=for_her")
        elif "unisex" in last_user_msg_lower or "both" in last_user_msg_lower or "either" in last_user_msg_lower:
            prefs["gender"] = "unisex"
            logger.info("node=collect_preferences direct_detect_gender=unisex")
        # If they just say generic words without specific gender indicator, try to detect
        elif any(word in last_user_msg_lower for word in ["men", "male", "him", "guy", "man"]) and "women" not in last_user_msg_lower:
            prefs["gender"] = "for_him"
            logger.info("node=collect_preferences direct_detect_gender=for_him (generic)")
        elif any(word in last_user_msg_lower for word in ["women", "female", "her", "lady", "woman"]) and "men" not in last_user_msg_lower:
            prefs["gender"] = "for_her"
            logger.info("node=collect_preferences direct_detect_gender=for_her (generic)")
    
    if awaiting_choice == "vibe" and not prefs.get("scent_types"):
        # User was asked about vibe/scent
        vibe_map = {
            "fresh": ["fresh", "breezy", "citrus", "clean"],
            "sweet": ["sweet", "gourmand", "vanilla", "honey"],
            "spicy": ["spicy", "bold", "warm"],
            "oud": ["oud", "intense", "habibi"],
            "floral": ["floral", "soft", "cozy", "rose"],
            "woody": ["woody", "earthy", "wood"],
        }
        for vibe_type, keywords in vibe_map.items():
            if any(kw in last_user_msg_lower for kw in keywords):
                prefs["scent_types"] = [vibe_type]
                logger.info("node=collect_preferences direct_detect_vibe=%s", vibe_type)
                break
    
    if awaiting_choice == "occasion" and not prefs.get("occasions"):
        # User was asked about occasion
        occasion_map = {
            "casual": ["daily", "everyday", "casual"],
            "formal": ["office", "work", "formal", "business", "wedding", "ceremony", "function", "event"],
            "date_night": ["date", "romantic", "date night"],
            "party": ["party", "night", "clubbing"],
            "mixed": ["mixed", "all", "everything", "any"],
        }
        for occ_type, keywords in occasion_map.items():
            if any(kw in last_user_msg_lower for kw in keywords):
                prefs["occasions"] = [occ_type]
                # Store raw occasion text for natural reference later
                for keyword in keywords:
                    if keyword in last_user_msg_lower:
                        prefs["occasion_raw"] = keyword
                        break
                logger.info("node=collect_preferences direct_detect_occasion=%s raw=%s", occ_type, prefs.get("occasion_raw"))
                break
    
    # Handle budget updates (both when awaiting and when user spontaneously changes)
    if awaiting_choice == "budget" or any(word in last_user_msg_lower for word in ["budget", "above", "under", "over", "₹"]):
        # User was asked about budget or is updating budget preference
        # Allow updates even if budget is already set (user can change their mind)
        budget_updated = False
        
        if "under" in last_user_msg_lower and "700" in last_user_msg_lower:
            prefs["budget"] = {"amount": 700, "operator": "under"}
            logger.info("node=collect_preferences direct_detect_budget=under_700")
            budget_updated = True
        elif ("above" in last_user_msg_lower or "over" in last_user_msg_lower or "more than" in last_user_msg_lower):
            # Extract amount dynamically
            import re
            amount_match = re.search(r'(\d+)', last_user_msg_lower)
            if amount_match:
                amount = int(amount_match.group(1))
                prefs["budget"] = {"amount": amount, "operator": "over"}
                logger.info("node=collect_preferences direct_detect_budget=above_%d", amount)
                budget_updated = True
        elif "700" in last_user_msg_lower and "900" in last_user_msg_lower:
            prefs["budget"] = {"amount": 800, "operator": "around"}
            logger.info("node=collect_preferences direct_detect_budget=700_900")
            budget_updated = True
        elif "no limit" in last_user_msg_lower or ("any" in last_user_msg_lower and ("budget" in last_user_msg_lower or awaiting_choice == "budget")):
            prefs["budget"] = None  # No budget constraint
            logger.info("node=collect_preferences direct_detect_budget=no_limit")
            budget_updated = True
        
        # Mark budget as asked if it was updated
        if budget_updated and "budget" not in state.get("questions_asked", []):
            questions_asked = state.get("questions_asked", [])
            questions_asked.append("budget")
            state["questions_asked"] = questions_asked

    # Use LLM to extract preferences from context (Smart extraction)
    try:
        llm_prompt = f"""{PREFERENCE_EXTRACTION_PROMPT}

Conversation Context:
{context_text}
"""
        llm_response = _call_llm(llm_prompt)
        # Clean up response to ensure it's valid JSON
        llm_response = llm_response.strip()
        if llm_response.startswith("```json"):
            llm_response = llm_response[7:]
        if llm_response.endswith("```"):
            llm_response = llm_response[:-3]
        
        import json
        extracted_prefs = json.loads(llm_response)
        
        # Merge extracted prefs into existing prefs (only update if not already set or explicitly changed)
        if extracted_prefs.get("gender"):
            # Only update gender if not already set
            if not prefs.get("gender"):
                prefs["gender"] = extracted_prefs["gender"]
        
        if extracted_prefs.get("scent_types"):
            existing_scents = prefs.get("scent_types", [])
            new_scents = extracted_prefs["scent_types"]
            # Only add new scents, don't replace existing ones
            prefs["scent_types"] = list(set(existing_scents + new_scents))
            
        if extracted_prefs.get("budget") and not budget_updated:
            # Only update budget from LLM if it wasn't already updated by direct detection above
            prefs["budget"] = extracted_prefs["budget"]
            
        if extracted_prefs.get("is_gift") is not None:
            # Only update is_gift if explicitly detected AND not already set
            # Preserve existing is_gift value unless user explicitly says it's (not) a gift
            if extracted_prefs["is_gift"] and not prefs.get("is_gift"):
                # User explicitly mentioned it's a gift, update it
                prefs["is_gift"] = True
            # Don't change is_gift to False if it was already True
            # User would need to explicitly say "not a gift" to change it

        if extracted_prefs.get("specific_notes"):
            existing_notes = prefs.get("specific_notes", [])
            new_notes = extracted_prefs["specific_notes"]
            prefs["specific_notes"] = list(set(existing_notes + new_notes))
            
        logger.info("node=collect_preferences llm_extraction_success=1 extracted=%s", extracted_prefs)
    except Exception as e:
        logger.error("node=collect_preferences llm_extraction_error=%s", e)

    # If we have a recommender module, let it do smarter extraction (Rule-based + Scoring)
    # Use context_text to extract preferences from entire conversation, not just last message
    if extract_preferences_and_recommend is not None:
        try:
            # Get recently recommended IDs to avoid repetition
            last_recs = state.get("last_recommendations") or []
            recent_ids = [p.get("id") for p in last_recs if isinstance(p, dict) and "id" in p]
            
            # Pass the updated prefs (with LLM extraction) to the recommender
            # Pass context_text for extraction, but last_user_msg for boosting
            rec_out = extract_preferences_and_recommend(
                context_text, 
                prefs, 
                latest_user_text=last_user_msg,
                recently_recommended_ids=recent_ids
            )
            # Support both tuple-style and dict-style return values
            if isinstance(rec_out, tuple):
                # recommender returned (new_prefs, recos)
                new_prefs, recos = rec_out
            else:
                # recommender returned {"preferences": {...}, "recommendations": [...]}
                new_prefs = rec_out.get("preferences") or prefs
                recos = rec_out.get("recommendations") or []
            # Update state
            state["preferences"] = new_prefs or prefs
            
            # Only set recommendations if we have sufficient preferences
            # For gift queries, need gender + vibe + budget (3 questions)
            # For regular queries, need ALL 4 questions asked
            is_gift = new_prefs.get("is_gift", False)
            questions_asked = state.get("questions_asked", [])
            required_questions = {"gender", "vibe", "occasion", "budget"}
            all_questions_asked = required_questions.issubset(set(questions_asked))
            
            if is_gift:
                gift_required_questions = {"gender", "vibe", "budget"}
                has_sufficient_prefs = gift_required_questions.issubset(set(questions_asked))
            else:
                has_sufficient_prefs = all_questions_asked
            
            if recos and has_sufficient_prefs:
                # pre-fill tool_result so tool_exec/response_node can use it directly
                state["tool_result"] = {"recommendations": recos}
            else:
                # Clear any pre-existing recommendations if we don't have enough preferences
                state["tool_result"] = None
            return state
        except Exception as e:
            logger.exception("node=collect_preferences error=%s", e)

    state["preferences"] = prefs
    return state


def tool_exec(state: ChatState):
    logger.info("node=tool_exec intent=%s", state.get("intent"))
    # if a previous node already put recommendations into tool_result, keep it
    if state.get("tool_result") and "recommendations" in state["tool_result"]:
        logger.info("node=tool_exec detected_pre_filled_recommendations=1")
        # Reset recommendation index when new recommendations are generated
        state["recommendation_index"] = 0
        return state

    intent = state["intent"]
    result = {}

    if state.get("missing_field") == "auth":
        state["tool_result"] = None
        logger.info("node=tool_exec blocked_missing_auth=1")
        return state

    if intent == "product_info":
        user_text = next(m["content"] for m in reversed(state["messages"]) if m["role"] == "user")
        prod = get_product_by_name(user_text)
        if prod:
            result = {"product": prod}
            logger.info("node=tool_exec path=product_info hit=exact")
        else:
            result = {"products": list_products()}
            logger.info("node=tool_exec path=product_info hit=list")
    elif intent in ("product_recommendation", "find_perfume", "gift_recommendation"):
        # If collect_preferences already set recommendations (with sufficient prefs), use them
        if state.get("tool_result") and isinstance(state.get("tool_result"), dict) and "recommendations" in state["tool_result"]:
            logger.info("node=tool_exec path=%s using_pre_filled_recommendations", intent)
            # Reset recommendation index when new recommendations are generated
            state["recommendation_index"] = 0
            return state
        
        # If collect_preferences set tool_result to None (insufficient prefs), preserve it
        if state.get("tool_result") is None:
            logger.info("node=tool_exec path=%s preserving_none_from_collect_prefs", intent)
            return state
        
        # Check if we have sufficient preferences before recommending
        prefs = state.get("preferences") or {}
        has_gender = prefs.get("gender") is not None
        has_scent_type = isinstance(prefs.get("scent_types"), list) and len(prefs.get("scent_types", [])) > 0
        has_budget = prefs.get("budget") is not None
        has_occasion = isinstance(prefs.get("occasions"), list) and len(prefs.get("occasions", [])) > 0
        is_gift = prefs.get("is_gift", False) or intent == "gift_recommendation"
        
        # Check if all required questions have been asked
        questions_asked = state.get("questions_asked", [])
        required_questions = {"gender", "vibe", "occasion", "budget"}
        all_questions_asked = required_questions.issubset(set(questions_asked))
        
        # For gift queries, need gender + vibe + budget (3 questions)
        # For regular queries, need ALL 4 questions asked
        if is_gift:
            gift_required_questions = {"gender", "vibe", "budget"}
            has_sufficient_prefs = gift_required_questions.issubset(set(questions_asked))
        else:
            has_sufficient_prefs = all_questions_asked
        
        # Only recommend if we have sufficient preferences
        if has_sufficient_prefs:
            user_text = next(m["content"] for m in reversed(state["messages"]) if m["role"] == "user")
            recs = recommend_products(user_text)
            result = {"recommendations": recs}
            # Reset recommendation index when new recommendations are generated
            state["recommendation_index"] = 0
            logger.info("node=tool_exec path=%s count=%d", intent, len(recs))
        else:
            # No sufficient preferences - don't set recommendations, response_node will ask questions
            result = None
            logger.info("node=tool_exec path=%s insufficient_prefs gender=%s scent=%s budget=%s is_gift=%s", 
                       intent, has_gender, has_scent_type, has_budget, is_gift)
    elif intent == "build_own_box":
        # Get BYOB products
        byob_products = get_byob_products()
        result = {"byob_products": byob_products}
        logger.info("node=tool_exec path=build_own_box count=%d", len(byob_products))
    elif intent == "general_question":
        user_text = next(m["content"] for m in reversed(state["messages"]) if m["role"] == "user")
        faq_answer = get_faq_answer(user_text)
        result = {"faq_answer": faq_answer}
        logger.info("node=tool_exec path=general_question found=%s", bool(faq_answer))
    elif intent == "returns_refunds":
        result = get_return_policy()
        logger.info("node=tool_exec path=returns_refunds")
    elif intent == "order_status":
        r = get_order_status(state["customer_email"], state["order_number"])
        result = r
        logger.info("node=tool_exec path=order_status error=%s", str(r.get("error")) if isinstance(r, dict) else "none")
    elif intent == "change_address":
        result = {"ok": False, "message": "Address change not allowed after dispatch."}
        logger.info("node=tool_exec path=change_address")
    else:
        result = {}
        logger.info("node=tool_exec path=unknown")

    state["tool_result"] = result
    return state


def response_node(state: ChatState):
    logger.info("node=response entering intent=%s missing=%s", state.get("intent"), state.get("missing_field"))
    user_text = next(m["content"] for m in reversed(state["messages"]) if m["role"] == "user")
    intent = state.get("intent")
    tool_result = state.get("tool_result")
    missing = state.get("missing_field")
    customer_authed = state.get("customer_authed", False)
    customer_email = state.get("customer_email")
    order_number = state.get("order_number")
    
    # Save awaiting_choice before clearing (needed for BYOB and preference flows)
    awaiting_choice_before_clear = state.get("awaiting_choice")
    
    # Clear button suggestions at start of response (will be regenerated if needed)
    state["button_suggestions"] = None
    # Don't clear awaiting_choice yet - we need it for BYOB and preference detection
    # It will be cleared or updated later in the flow

    # Handle human handoff flow
    handoff_state = state.get("handoff_state")
    if intent == "human_handoff" or handoff_state:
        # Force intent to human_handoff if we're in handoff flow
        if handoff_state:
            state["intent"] = "human_handoff"
        
        # Extract email/phone from user message if provided
        user_text_lower = user_text.lower()
        email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", user_text)
        phone_match = re.search(r"[\d\s\-\+\(\)]{10,}", user_text)
        
        if handoff_state == "collecting_email":
            # User is providing email
            if email_match:
                state["handoff_email"] = email_match.group(0)
                state["handoff_state"] = "collecting_phone"
                reply = "Thank you! Please provide your phone number so we can contact you."
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response handoff_email_collected=%s", state["handoff_email"])
                return state
            else:
                reply = "I didn't catch a valid email address. Please provide your email address."
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response handoff_email_invalid=1")
                return state
        elif handoff_state == "collecting_phone":
            # User is providing phone
            if phone_match:
                state["handoff_phone"] = phone_match.group(0).strip()
                # Generate ticket (in real implementation, this would call a ticket system API)
                ticket_id = f"TKT-{int(time.time())}"
                state["handoff_state"] = None  # Clear handoff state to allow normal conversation
                state["handoff_completed"] = True  # Keep flag for reference but don't block
                reply = f"Ok, we have raised the ticket (Ticket ID: {ticket_id}). Someone will contact you shortly at {state['handoff_email']} or {state['handoff_phone']}. Can I help you with anything else?"
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response handoff_completed ticket_id=%s email=%s phone=%s", 
                          ticket_id, state["handoff_email"], state["handoff_phone"])
                return state
            else:
                reply = "I didn't catch a valid phone number. Please provide your phone number."
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response handoff_phone_invalid=1")
                return state
        elif handoff_state == "awaiting_confirmation":
            # User is confirming they want to talk to human
            user_text_lower = user_text.lower()
            if any(word in user_text_lower for word in ["yes", "yeah", "yep", "sure", "ok", "okay", "confirm", "proceed", "connect me"]) or "confirm_handoff" in user_text:
                state["handoff_state"] = "collecting_email"
                reply = "Great! To connect you with a human agent, I'll need your email address first."
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response handoff_confirmed=1")
                return state
            elif any(word in user_text_lower for word in ["no", "nope", "cancel", "cancel_handoff"]) or "cancel_handoff" in user_text:
                # User declined or changed mind
                state["handoff_state"] = None
                reply = "No problem! How else can I help you today?"
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response handoff_declined=1")
                return state
        else:
            # First time - ask for confirmation
            state["handoff_state"] = "awaiting_confirmation"
            reply = "I understand you'd like to speak with a human agent. Would you like me to connect you with one?"
            buttons = [
                {"label": "Yes", "action": "confirm_handoff", "type": "primary"},
                {"label": "No", "action": "cancel_handoff", "type": "secondary"},
            ]
            state["button_suggestions"] = buttons
            state["messages"].append({"role": "assistant", "content": reply})
            logger.info("node=response handoff_initiated=1")
            return state

    if missing == "auth":
        reply = (
            "I can look up your order. Please share the email you used and the order number (like BLB-1001). "
            "After that, I'll tell you the exact status. Can I help you with anything else?"
        )
        state["messages"].append({"role": "assistant", "content": reply})
        logger.info("node=response auth_prompted=1")
        return state

    # Check if user just provided auth details and we should fetch order status
    # This handles the case where user provides email/order but intent wasn't detected as order_status
    if customer_authed and customer_email and order_number:
        # Check if user is asking about order status (current intent or in conversation history)
        # Look at recent user messages to see if they asked about order status
        recent_user_messages = [m.get("content", "").lower() for m in state["messages"][-5:] if m.get("role") == "user"]
        is_order_related = (
            intent == "order_status" or 
            "order" in user_text.lower() or 
            ("where" in user_text.lower() and "order" in user_text.lower()) or
            any("order" in msg or ("where" in msg and "order" in msg) for msg in recent_user_messages)
        )
        
        # If we have auth but no tool_result yet (or error), and user is asking about orders, fetch it now
        if is_order_related and (not tool_result or (isinstance(tool_result, dict) and tool_result.get("error"))):
            logger.info("node=response fetching_order_status_post_auth email=%s order=%s", customer_email, order_number)
            from tools import get_order_status
            tool_result = get_order_status(customer_email, order_number)
            state["tool_result"] = tool_result
            # Update intent to order_status if it wasn't already
            if intent != "order_status":
                state["intent"] = "order_status"

    # For order_status, compose a direct concise message including items and shipping
    if intent == "order_status" and isinstance(tool_result, dict) and not tool_result.get("error"):
        items = tool_result.get("items", [])
        items_str = ", ".join([f"{it.get('name')} (Qty: {it.get('quantity', 1)})" for it in items]) if items else ""
        parts = []
        parts.append(f"Order {tool_result.get('order_number')} is {tool_result.get('status')}")
        if items_str:
            parts.append(f"Items: {items_str}")
        if tool_result.get("carrier") and tool_result.get("tracking_id"):
            parts.append(f"Shipped via {tool_result.get('carrier')} (Tracking: {tool_result.get('tracking_id')})")
        if tool_result.get("eta"):
            parts.append(f"ETA: {tool_result.get('eta')}")
        if tool_result.get("shipping_address"):
            parts.append(f"Ship-to: {tool_result.get('shipping_address')}")
        reply = ". ".join(parts) + ".\n\nCan I help you with anything else?"
        state["messages"].append({"role": "assistant", "content": reply})
        logger.info("node=response composed_order_status reply_len=%d", len(reply))
        return state
    
    # Handle order_status errors
    if intent == "order_status" and isinstance(tool_result, dict) and tool_result.get("error"):
        error = tool_result.get("error")
        if error == "customer_not_found":
            reply = "I couldn't find an account with that email. Please check and try again. Can I help you with anything else?"
        elif error == "order_not_found":
            reply = f"I couldn't find order {order_number} for that email. Please verify your order number and try again. Can I help you with anything else?"
        else:
            reply = "I'm having trouble looking up your order right now. Please try again later. Can I help you with anything else?"
        state["messages"].append({"role": "assistant", "content": reply})
        logger.info("node=response order_status_error=%s", error)
        return state

    # Handle questions about recommended products
    if intent == "product_question":
        last_recs = state.get("last_recommendations", [])
        if last_recs:
            # Extract information directly from product data to avoid LLM hallucination
            user_text_lower = user_text.lower()
            
            # Check if user is asking for NEW recommendations based on specific notes or product type
            # Patterns like "with lemon", "anything with rose", "single product not package"
            note_request_patterns = [
                "with ", "anything with", "something with", "perfume with",
                "that has", "which has", "containing"
            ]
            is_note_request = any(pattern in user_text_lower for pattern in note_request_patterns)
            
            # Check if user is changing product type (single product vs package/gift set)
            single_product_patterns = [
                "single product", "single perfume", "single bottle", "individual product",
                "not package", "no package", "not a package", "not collection", "no collection"
            ]
            package_patterns = [
                "package", "collection", "gift set", "set", "bundle"
            ]
            is_single_product_request = any(pattern in user_text_lower for pattern in single_product_patterns)
            is_package_request = any(pattern in user_text_lower for pattern in package_patterns) and not is_single_product_request
            
            # If user is asking for products with specific notes or changing product type, treat as new product_recommendation
            if is_note_request or is_single_product_request or is_package_request:
                # Re-route to product_recommendation flow to find new matches
                state["intent"] = "product_recommendation"
                logger.info("node=response re_routing_note_request_to_recommendation")
                # Let the recommendation flow handle this
                # Clear awaiting_choice to start fresh
                state["awaiting_choice"] = None
                # Continue to product_recommendation flow below
                intent = "product_recommendation"
            else:
                # Try to find product by name if mentioned - check both in last_recs and catalog
                product = None
                
                # First, check if user is asking about a product in last_recs
                for rec in last_recs:
                    rec_name = rec.get("name", "").lower()
                    # Remove size info for better matching
                    rec_name_clean = rec_name.replace(" - 100ml", "").replace(" - 50ml", "").strip()
                    # Get significant words (length > 2, not common words)
                    rec_words = [w for w in rec_name_clean.split() if len(w) > 2 and w not in ["the", "and", "for"]]
                    
                    # Check if significant words from product name appear in query
                    if rec_words and any(word in user_text_lower for word in rec_words):
                        product = rec
                        logger.info("node=response matched_product_from_last_recs product_name=%s", rec.get("name"))
                        break
                
                # If no match in last_recs, try searching the entire catalog
                # This handles cases where user asks about a different product
                if not product:
                    catalog_product = get_product_by_name(user_text)
                    if catalog_product:
                        # Check if this is a different product than what's in last_recs
                        catalog_id = catalog_product.get("id")
                        is_different_product = not any(rec.get("id") == catalog_id for rec in last_recs)
                        
                        if is_different_product:
                            logger.info("node=response found_different_product_in_catalog product_name=%s", catalog_product.get("name"))
                            # User is asking about a different product - update last_recommendations
                            state["last_recommendations"] = [catalog_product]
                            last_recs = [catalog_product]
                            product = catalog_product
                        else:
                            # Same product, use it
                            product = catalog_product
                            logger.info("node=response found_same_product_in_catalog product_name=%s", catalog_product.get("name"))
                
                # If still no specific product found, use first one from last_recs
                if not product:
                    product = last_recs[0] if last_recs else {}
                    logger.info("node=response using_first_from_last_recs product_name=%s", product.get("name"))
                
                # Handle price questions
                if any(word in user_text_lower for word in ["price", "cost", "how much"]):
                    price = product.get("price", 0)
                    name = product.get("name", "this product")
                    if price:
                        reply = f"The price of {name} is ₹{price}. Want to see it or explore other options?"
                    else:
                        reply = f"I don't have the price information for {name} right now. Can I help you with anything else?"
                    
                    # Add action buttons
                    buttons = []
                    product_url = product.get("product_url", "")
                    if product_url:
                        buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
                    
                    # Add other options button if more recommendations available
                    current_index = state.get("recommendation_index", 0)
                    remaining = len(last_recs) - current_index - 1
                    if remaining > 0:
                        buttons.append({"label": "Other Options", "action": "show_other", "type": "primary"})
                    
                    # Add safer/bolder buttons
                    current_price = product.get("price", 0)
                    safer_exists = any(p.get("price", 0) < current_price for p in last_recs if p.get("id") != product.get("id"))
                    if safer_exists:
                        buttons.append({"label": "Show Safer Option", "action": "show_safer", "type": "secondary"})
                    
                    bolder_exists = any(p.get("price", 0) > current_price for p in last_recs if p.get("id") != product.get("id"))
                    if bolder_exists:
                        buttons.append({"label": "Show Bolder Option", "action": "show_bolder", "type": "secondary"})
                    
                    if buttons:
                        state["button_suggestions"] = buttons
                    
                    state["messages"].append({"role": "assistant", "content": reply})
                    logger.info("node=response answered_price_question=1 buttons=%d", len(buttons))
                    return state
                
                # Handle occasion/when to wear questions
                if any(phrase in user_text_lower for phrase in ["when", "wear", "occasion", "use", "suitable"]):
                    name = product.get("name", "this product")
                    occasion_tags = product.get("occasion_tags", [])
                    vibe_tags = product.get("vibe_tags", [])
                    
                    reply_parts = [f"{name} is great for"]
                    
                    if occasion_tags:
                        # Format occasions nicely
                        occasions = [occ.replace("_", " ") for occ in occasion_tags]
                        if len(occasions) == 1:
                            reply_parts.append(occasions[0])
                        elif len(occasions) == 2:
                            reply_parts.append(f"{occasions[0]} and {occasions[1]}")
                        else:
                            reply_parts.append(f"{', '.join(occasions[:-1])}, and {occasions[-1]}")
                    else:
                        reply_parts.append("everyday wear")
                    
                    # Add vibe context
                    if vibe_tags:
                        vibe_desc = ", ".join(vibe_tags[:3])
                        reply_parts.append(f"It has {vibe_desc} vibes.")
                    
                    reply = " ".join(reply_parts) + " Want to see it or explore other options?"
                    
                    # Add action buttons
                    buttons = []
                    product_url = product.get("product_url", "")
                    if product_url:
                        buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
                    
                    # Add other options button if more recommendations available
                    current_index = state.get("recommendation_index", 0)
                    remaining = len(last_recs) - current_index - 1
                    if remaining > 0:
                        buttons.append({"label": "Other Options", "action": "show_other", "type": "primary"})
                    
                    # Add safer/bolder buttons
                    current_price = product.get("price", 0)
                    current_vibes = product.get("vibe_tags", [])
                    
                    safer_candidates = [
                        p for p in last_recs 
                        if p.get("id") != product.get("id") and (
                            p.get("price", 0) < current_price or
                            any(v in p.get("vibe_tags", []) for v in ["fresh", "clean", "light", "breezy"])
                        )
                    ]
                    if safer_candidates:
                        buttons.append({"label": "Show Safer Option", "action": "show_safer", "type": "secondary"})
                    
                    bolder_candidates = [
                        p for p in last_recs 
                        if p.get("id") != product.get("id") and (
                            p.get("price", 0) > current_price or
                            any(v in p.get("vibe_tags", []) for v in ["bold", "intense", "strong", "oud", "spicy"])
                        )
                    ]
                    if bolder_candidates:
                        buttons.append({"label": "Show Bolder Option", "action": "show_bolder", "type": "secondary"})
                    
                    if buttons:
                        state["button_suggestions"] = buttons
                    
                    state["messages"].append({"role": "assistant", "content": reply})
                    logger.info("node=response answered_occasion_question=1 buttons=%d", len(buttons))
                    return state
                
                # Handle questions about contents/what's included
                if any(phrase in user_text_lower for phrase in ["what do i get", "what's in", "what is in", "what comes", "contents", "includes"]):
                    # Use the product found above, or first one
                    if not product:
                        product = last_recs[0] if last_recs else {}
                    name = product.get("name", "this product")
                    desc = product.get("short_description", "")
                    notes = product.get("notes", {})
                    
                    reply_parts = [f"{name}"]
                    if desc:
                        reply_parts.append(f"includes: {desc}")
                    
                    # Extract notes if available
                    all_notes = []
                    if notes:
                        all_notes = (
                            notes.get("head", []) + 
                            notes.get("heart", []) + 
                            notes.get("base", [])
                        )
                    
                    if all_notes:
                        notes_str = ", ".join(all_notes[:5])  # Limit to first 5 notes
                        reply_parts.append(f"Key notes: {notes_str}")
                    
                    reply = ". ".join(reply_parts) + ". Want to see it or explore other options?"
                    
                    # Add action buttons
                    buttons = []
                    product_url = product.get("product_url", "")
                    if product_url:
                        buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
                    
                    current_index = state.get("recommendation_index", 0)
                    remaining = len(last_recs) - current_index - 1
                    if remaining > 0:
                        buttons.append({"label": "Other Options", "action": "show_other", "type": "primary"})
                    
                    if buttons:
                        state["button_suggestions"] = buttons
                    
                    state["messages"].append({"role": "assistant", "content": reply})
                    logger.info("node=response answered_contents_question=1 buttons=%d", len(buttons))
                    return state
                
                # Handle other questions - use LLM but with structured product data
                import json
                products_context = json.dumps(last_recs[:2], indent=2)
                
                # Get conversation context (recent user messages for occasion/context)
                user_messages = [m["content"] for m in state["messages"] if m["role"] == "user"]
                conversation_context = " ".join(user_messages[-3:]) if len(user_messages) > 1 else ""
                
                # Get user preferences for context
                prefs = state.get("preferences") or {}
                occasion_context = ""
                if prefs.get("occasion_raw"):
                    # Use raw occasion text (e.g., "wedding") for natural reference
                    occasion_context = f"\nUser mentioned occasion: {prefs.get('occasion_raw')}"
                elif prefs.get("occasions"):
                    occasions = prefs.get("occasions", [])
                    occasion_context = f"\nUser mentioned occasion: {', '.join(occasions)}"
                
                prompt = f"""{ASSISTANT_STYLE_PROMPT}

The user is asking about these recommended products:
{products_context}

Conversation context: {conversation_context}
{occasion_context}

User question: {user_text}

IMPORTANT: 
- Extract information directly from the product data above. Do not make up prices or details.
- Price is in the "price" field (in rupees ₹)
- Description is in "short_description"
- Notes are in "notes" object with "head", "heart", "base" arrays
- Name is in "name" field
- Reference the conversation context (like wedding, occasion, etc.) in your answer to make it relevant
- DO NOT use markdown formatting, asterisks (**), or bold text - write in plain text only
- Keep response SHORT (2-3 sentences max) and conversational
- If user mentioned specific occasion earlier (wedding, date, party), reference it naturally

Answer their question naturally using ONLY the data provided above.
End with: "Want to see it or explore other options?"
"""
                answer = _call_llm(prompt)
                
                # Add action buttons
                buttons = []
                current_index = state.get("recommendation_index", 0)
                if current_index < len(last_recs):
                    product = last_recs[current_index]
                    product_url = product.get("product_url", "")
                    if product_url:
                        buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
                
                remaining = len(last_recs) - current_index - 1
                if remaining > 0:
                    buttons.append({"label": "Other Options", "action": "show_other", "type": "primary"})
                
                if buttons:
                    state["button_suggestions"] = buttons
                
                state["messages"].append({"role": "assistant", "content": answer})
                logger.info("node=response answered_product_question=1 buttons=%d", len(buttons))
                return state
        else:
            # No last recommendations - try to find product in catalog by name
            product = get_product_by_name(user_text)
            if product:
                logger.info("node=response found_product_in_catalog product_name=%s", product.get("name"))
                # Add to last_recommendations so it can be referenced
                state["last_recommendations"] = [product]
                last_recs = [product]
                
                # Extract information directly from product data
                user_text_lower = user_text.lower()
                
                # Handle price questions
                if any(word in user_text_lower for word in ["price", "cost", "how much"]):
                    price = product.get("price", 0)
                    name = product.get("name", "this product")
                    if price:
                        reply = f"The price of {name} is ₹{price}. Want to see it or explore similar options?"
                    else:
                        reply = f"I don't have the price information for {name} right now. Can I help you with anything else?"
                    
                    # Add action buttons
                    buttons = []
                    product_url = product.get("product_url", "")
                    if product_url:
                        buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
                    
                    buttons.append({"label": "Find My Perfume", "action": "find_perfume", "type": "primary"})
                    
                    if buttons:
                        state["button_suggestions"] = buttons
                    
                    state["messages"].append({"role": "assistant", "content": reply})
                    logger.info("node=response answered_price_question_from_catalog=1 buttons=%d", len(buttons))
                    return state
                
                # Handle occasion/when to wear questions
                if any(phrase in user_text_lower for phrase in ["when", "wear", "occasion", "use", "suitable"]):
                    name = product.get("name", "this product")
                    occasion_tags = product.get("occasion_tags", [])
                    vibe_tags = product.get("vibe_tags", [])
                    
                    reply_parts = [f"{name} is great for"]
                    
                    if occasion_tags:
                        # Format occasions nicely
                        occasions = [occ.replace("_", " ") for occ in occasion_tags]
                        if len(occasions) == 1:
                            reply_parts.append(occasions[0])
                        elif len(occasions) == 2:
                            reply_parts.append(f"{occasions[0]} and {occasions[1]}")
                        else:
                            reply_parts.append(f"{', '.join(occasions[:-1])}, and {occasions[-1]}")
                    else:
                        reply_parts.append("everyday wear")
                    
                    # Add vibe context
                    if vibe_tags:
                        vibe_desc = ", ".join(vibe_tags[:3])
                        reply_parts.append(f"It has {vibe_desc} vibes.")
                    
                    reply = " ".join(reply_parts) + " Want to see it or explore similar options?"
                    
                    # Add action buttons
                    buttons = []
                    product_url = product.get("product_url", "")
                    if product_url:
                        buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
                    
                    buttons.append({"label": "Find My Perfume", "action": "find_perfume", "type": "primary"})
                    
                    if buttons:
                        state["button_suggestions"] = buttons
                    
                    state["messages"].append({"role": "assistant", "content": reply})
                    logger.info("node=response answered_occasion_question_from_catalog=1 buttons=%d", len(buttons))
                    return state
                
                # Handle questions about contents/what's included/notes
                if any(phrase in user_text_lower for phrase in ["what do i get", "what's in", "what is in", "what comes", "contents", "includes", "notes", "what notes"]):
                    name = product.get("name", "this product")
                    desc = product.get("short_description", "")
                    notes = product.get("notes", {})
                    
                    reply_parts = [f"{name}"]
                    if desc:
                        reply_parts.append(f"includes: {desc}")
                    
                    # Extract notes if available
                    all_notes = []
                    if notes:
                        all_notes = (
                            notes.get("head", []) + 
                            notes.get("heart", []) + 
                            notes.get("base", [])
                        )
                    
                    if all_notes:
                        notes_str = ", ".join(all_notes[:5])  # Limit to first 5 notes
                        reply_parts.append(f"Key notes: {notes_str}")
                    
                    reply = ". ".join(reply_parts) + ". Want to see it or explore similar options?"
                    
                    # Add action buttons
                    buttons = []
                    product_url = product.get("product_url", "")
                    if product_url:
                        buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
                    
                    buttons.append({"label": "Find My Perfume", "action": "find_perfume", "type": "primary"})
                    
                    if buttons:
                        state["button_suggestions"] = buttons
                    
                    state["messages"].append({"role": "assistant", "content": reply})
                    logger.info("node=response answered_contents_question_from_catalog=1 buttons=%d", len(buttons))
                    return state
                
                # For other questions, use LLM with product data
                import json
                product_context = json.dumps(product, indent=2)
                
                prompt = f"""{ASSISTANT_STYLE_PROMPT}

The user is asking about this product:
{product_context}

User question: {user_text}

IMPORTANT: 
- Extract information directly from the product data above. Do not make up prices or details.
- Price is in the "price" field (in rupees ₹)
- Description is in "short_description"
- Notes are in "notes" object with "head", "heart", "base" arrays
- Name is in "name" field
- DO NOT use markdown formatting, asterisks (**), or bold text - write in plain text only
- Keep response SHORT (2-3 sentences max) and conversational

Answer their question naturally using ONLY the data provided above.
End with: "Want to see it or explore similar options?"
"""
                answer = _call_llm(prompt)
                
                # Add action buttons
                buttons = []
                product_url = product.get("product_url", "")
                if product_url:
                    buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
                
                buttons.append({"label": "Find My Perfume", "action": "find_perfume", "type": "primary"})
                
                if buttons:
                    state["button_suggestions"] = buttons
                
                state["messages"].append({"role": "assistant", "content": answer})
                logger.info("node=response answered_product_question_from_catalog=1 buttons=%d", len(buttons))
                return state
            else:
                # Product not found in catalog either
                reply = "I don't have any products in context right now. Could you tell me what you're looking for? Can I help you with anything else?"
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response product_question_no_context=1")
                return state

    # Handle BYOB flow
    if intent == "build_own_box":
        if isinstance(tool_result, dict) and tool_result.get("byob_products"):
            byob_products = tool_result["byob_products"]
            
            # Check if user has already started selecting
            selections = state.get("byob_selections") or []
            max_items = state.get("byob_max_items", 3)
            
            # Check if we're awaiting a BYOB selection and user just selected a product
            # Use the saved value from before clearing
            awaiting_byob = awaiting_choice_before_clear == "byob_selection"
            logger.info("node=response byob_check awaiting=%s user_text=%s selections=%s", awaiting_byob, user_text, selections)
            
            if awaiting_byob:
                # Try to match user's message to a BYOB product name
                user_text_lower = user_text.lower().strip()
                selected_product_id = None
                
                # First check if message contains "select_byob:" action format
                if "select_byob:" in user_text:
                    try:
                        selected_product_id = user_text.split("select_byob:")[1].strip()
                        logger.info("node=response byob_found_action_format product_id=%s", selected_product_id)
                    except Exception as e:
                        logger.error("node=response byob_action_parse_error error=%s", e)
                        pass
                
                # If not found, try matching by product name
                if not selected_product_id:
                    logger.info("node=response byob_matching_by_name user_text_lower=%s products_count=%d", user_text_lower, len(byob_products))
                    for product in byob_products:
                        product_name = product.get("name", "").lower().strip()
                        product_id = product.get("id", "")
                        # More flexible matching: check if user text matches product name
                        # Remove common punctuation and extra spaces for better matching
                        product_name_clean = product_name.replace("-", " ").replace("  ", " ")
                        user_text_clean = user_text_lower.replace("-", " ").replace("  ", " ")
                        
                        # Check multiple matching strategies
                        if (product_name == user_text_lower or 
                            user_text_lower == product_name or
                            user_text_clean == product_name_clean or
                            user_text_lower in product_name or 
                            product_name in user_text_lower or
                            user_text_clean in product_name_clean or
                            product_name_clean in user_text_clean):
                            selected_product_id = product_id
                            logger.info("node=response byob_matched_product product_id=%s product_name=%s", product_id, product_name)
                            break
                
                if not selected_product_id:
                    logger.warning("node=response byob_no_match user_text=%s", user_text)
                
                # If we found a selection and it's not already selected
                if selected_product_id and selected_product_id not in selections:
                    selections.append(selected_product_id)
                    state["byob_selections"] = selections
                    logger.info("node=response byob_product_selected=1 product_id=%s total_selected=%d", selected_product_id, len(selections))
                    
                    # Check if we've reached the max
                    if len(selections) >= max_items:
                        # User has completed selection
                        validation = validate_byob_selection(selections, max_items)
                        if validation.get("valid"):
                            selected_products = validation["products"]
                            total_price = validation["total_price"]
                            
                            # Build summary
                            product_names = [p.get("name", "") for p in selected_products]
                            reply = f"Perfect! Your custom box includes: {', '.join(product_names)}. "
                            reply += f"Total: ₹{total_price}. "
                            reply += "Visit our BYOB page to complete your order. Can I help you with anything else?"
                            
                            # Add BYOB page link button
                            buttons = [{"label": "View BYOB Page", "action": "open_url:https://blabliblulife.com/pages/build-your-own-box", "type": "success"}]
                            state["button_suggestions"] = buttons
                        else:
                            reply = "There was an issue with your selections. Let's start over. Can I help you with anything else?"
                        
                        state["messages"].append({"role": "assistant", "content": reply})
                        # Clear selections after completion
                        state["byob_selections"] = []
                        state["awaiting_choice"] = None
                        logger.info("node=response byob_completed=1")
                        return state
                    else:
                        # Still need more selections - show updated list
                        remaining = max_items - len(selections)
                        reply = f"Great! You've selected {len(selections)}. Choose {remaining} more. "
                elif selected_product_id in selections:
                    # Product already selected
                    remaining = max_items - len(selections)
                    reply = f"You've already selected that one. You've selected {len(selections)}. Choose {remaining} more. "
                else:
                    # Couldn't match the selection - show products again
                    remaining = max_items - len(selections)
                    if len(selections) == 0:
                        reply = f"Great! You can choose {max_items} perfumes for your custom box. "
                    else:
                        reply = f"You've selected {len(selections)}. Choose {remaining} more. "
            else:
                # First time showing products
                remaining = max_items - len(selections)
                if len(selections) == 0:
                    reply = f"Great! You can choose {max_items} perfumes for your custom box. "
                else:
                    reply = f"You've selected {len(selections)}. Choose {remaining} more. "
            
            # Generate button suggestions for selection (exclude already selected)
            buttons = []
            for product in byob_products[:8]:  # Show top 8 options
                pid = product.get("id", "")
                name = product.get("name", "")
                if pid not in selections:
                    buttons.append({
                        "label": name,
                        "action": f"select_byob:{pid}",
                        "type": "primary"
                    })
            
            state["button_suggestions"] = buttons
            state["awaiting_choice"] = "byob_selection"
            
            reply += "Select your perfumes from the options below."
            state["messages"].append({"role": "assistant", "content": reply})
            logger.info("node=response byob_awaiting_selection remaining=%d selected=%d", remaining, len(selections))
            return state
        else:
            reply = "I couldn't load the BYOB products right now. Please try again. Can I help you with anything else?"
            state["messages"].append({"role": "assistant", "content": reply})
            logger.info("node=response byob_error=no_products")
            return state
    
    # Handle general questions (FAQ)
    if intent == "general_question":
        if isinstance(tool_result, dict) and tool_result.get("faq_answer"):
            reply = tool_result["faq_answer"] + " Can I help you with anything else?"
            state["messages"].append({"role": "assistant", "content": reply})
            logger.info("node=response general_question_answered=1")
            return state
        else:
            # No FAQ match - offer human handoff
            reply = "I'm not sure about that. Would you like to talk with a human agent who can help you better?"
            buttons = [
                {"label": "Yes, connect me", "action": "confirm_handoff", "type": "primary"},
                {"label": "No, thanks", "action": "cancel_handoff", "type": "secondary"},
            ]
            state["button_suggestions"] = buttons
            state["handoff_state"] = "awaiting_confirmation"
            state["messages"].append({"role": "assistant", "content": reply})
            logger.info("node=response general_question_no_match offering_handoff=1")
            return state
    
    # For recommendations: ask preferences one by one, remember from context
    if intent in ("product_recommendation", "find_perfume", "gift_recommendation"):
        # Check if we have sufficient preferences before recommending
        prefs = state.get("preferences") or {}
        has_gender = prefs.get("gender") is not None
        has_scent_type = isinstance(prefs.get("scent_types"), list) and len(prefs.get("scent_types", [])) > 0
        has_budget = prefs.get("budget") is not None
        
        # If intent is gift_recommendation, mark as gift
        is_gift = prefs.get("is_gift", False) or intent == "gift_recommendation"
        if intent == "gift_recommendation" and not prefs.get("is_gift"):
            prefs["is_gift"] = True
            state["preferences"] = prefs
        
        # Initialize questions_asked if not present
        if "questions_asked" not in state:
            state["questions_asked"] = []
        
        questions_asked = state.get("questions_asked", [])
        
        # If gender was auto-detected (e.g., from "girlfriend", "boyfriend"), mark it as asked
        if has_gender and "gender" not in questions_asked:
            questions_asked.append("gender")
            state["questions_asked"] = questions_asked
        
        has_occasion = isinstance(prefs.get("occasions"), list) and len(prefs.get("occasions", [])) > 0
        
        # For gift queries, follow the flow from bot_suggestion_flow.txt:
        # Ask gender -> vibe (personality) -> budget (3 questions before recommending)
        # For regular queries, follow the flow from bot_suggestion_flow.txt:
        # Ask gender -> vibe -> occasion -> budget (all 4 questions before recommending)
        if is_gift:
            # Gifts need: gender, vibe (personality), budget
            # Check if these specific questions have been asked
            gift_required_questions = {"gender", "vibe", "budget"}
            has_sufficient_prefs = gift_required_questions.issubset(set(questions_asked))
        else:
            # Only recommend after ALL 4 questions have been asked
            # Questions: gender, vibe, occasion, budget
            required_questions = {"gender", "vibe", "occasion", "budget"}
            has_sufficient_prefs = required_questions.issubset(set(questions_asked))
        
        # Only recommend if we have sufficient preferences AND recommendations exist
        if has_sufficient_prefs:
            if isinstance(tool_result, dict) and tool_result.get("recommendations"):
                recs = tool_result["recommendations"]
                
                # Check what the user is asking for
                user_text_lower = user_text.lower()
                asking_for_safer = any(phrase in user_text_lower for phrase in ["safer option", "safer", "lighter", "softer"])
                asking_for_bolder = any(phrase in user_text_lower for phrase in ["bolder option", "bolder", "stronger", "more intense"])
                asking_for_other = any(phrase in user_text_lower for phrase in [
                    "other option", "another option", "other options", "anything else", "something else",
                    "show me another", "different one", "what else", "more options"
                ])
                
                # Get current recommendation index (default to 0 if not set)
                current_index = state.get("recommendation_index", 0)
                current_product = recs[current_index] if current_index < len(recs) else None
                
                # Handle safer/bolder/other options
                # Get user's gender preference to filter recommendations
                user_gender = prefs.get("gender")
                
                if asking_for_safer and current_product:
                    # Find the safest option (lowest price or lightest vibe) matching user's gender
                    current_price = current_product.get("price", 0)
                    safer_recs = [
                        (idx, p) for idx, p in enumerate(recs)
                        if idx != current_index and (
                            # Match gender if specified, or accept unisex
                            (not user_gender or p.get("gender_profile") == user_gender or p.get("gender_profile") == "unisex")
                        ) and (
                            p.get("price", 0) < current_price or
                            any(v in p.get("vibe_tags", []) for v in ["fresh", "clean", "light", "breezy"])
                        )
                    ]
                    if safer_recs:
                        # Sort by price (ascending) and pick the first
                        safer_recs.sort(key=lambda x: x[1].get("price", 0))
                        current_index = safer_recs[0][0]
                elif asking_for_bolder and current_product:
                    # Find the boldest option (highest price or strongest vibe) matching user's gender
                    current_price = current_product.get("price", 0)
                    bolder_recs = [
                        (idx, p) for idx, p in enumerate(recs)
                        if idx != current_index and (
                            # Match gender if specified, or accept unisex
                            (not user_gender or p.get("gender_profile") == user_gender or p.get("gender_profile") == "unisex")
                        ) and (
                            p.get("price", 0) > current_price or
                            any(v in p.get("vibe_tags", []) for v in ["bold", "intense", "strong", "oud", "spicy"])
                        )
                    ]
                    if bolder_recs:
                        # Sort by price (descending) and pick the first
                        bolder_recs.sort(key=lambda x: x[1].get("price", 0), reverse=True)
                        current_index = bolder_recs[0][0]
                elif asking_for_other:
                    # Just show the next product in the list
                    current_index += 1
                
                # Check if we have more recommendations to show
                if current_index < len(recs):
                    top = recs[current_index]
                    state["last_recommendations"] = recs  # Keep all recommendations
                    state["recommendation_index"] = current_index  # Update index
                    
                    name = top.get("name", "")
                    positioning = top.get("positioning_line", "")
                    desc = top.get("short_description", "")
                    price = top.get("price", 0)
                    
                    # Build a natural response with context awareness
                    reply_parts = []
                    
                    # Add context prefix based on what user selected
                    if asking_for_other:
                        reply_parts.append(f"Here's another option: {name}.")
                    elif asking_for_bolder:
                        reply_parts.append(f"Here's a bolder option: {name}.")
                    elif asking_for_safer:
                        reply_parts.append(f"Here's a safer option: {name}.")
                    else:
                        reply_parts.append(f"I'd suggest {name}.")
                    
                    # Keep response short - combine positioning and description
                    if positioning:
                        reply_parts.append(positioning)
                    elif desc:
                        reply_parts.append(desc)
                    
                    if price:
                        reply_parts.append(f"Priced at ₹{price}.")
                    
                    # Add button suggestions for safer/bolder options and other actions
                    buttons = []
                    remaining = len(recs) - current_index - 1
                    
                    # Add safer/bolder option buttons based on product characteristics
                    # Safer = lower price or lighter scent, Bolder = higher price or stronger scent
                    current_price = top.get("price", 0)
                    current_vibes = top.get("vibe_tags", [])
                    
                    # Find safer option (lighter/cheaper) matching user's gender
                    safer_candidates = [
                        (idx, p) for idx, p in enumerate(recs) 
                        if idx != current_index and (
                            # Match gender if specified, or accept unisex
                            (not user_gender or p.get("gender_profile") == user_gender or p.get("gender_profile") == "unisex")
                        ) and (
                            p.get("price", 0) < current_price or
                            any(v in p.get("vibe_tags", []) for v in ["fresh", "clean", "light", "breezy"])
                        )
                    ]
                    if safer_candidates:
                        buttons.append({"label": "Show Safer Option", "action": "show_safer", "type": "secondary"})
                    
                    # Find bolder option (stronger/pricier) matching user's gender
                    bolder_candidates = [
                        (idx, p) for idx, p in enumerate(recs) 
                        if idx != current_index and (
                            # Match gender if specified, or accept unisex
                            (not user_gender or p.get("gender_profile") == user_gender or p.get("gender_profile") == "unisex")
                        ) and (
                            p.get("price", 0) > current_price or
                            any(v in p.get("vibe_tags", []) for v in ["bold", "intense", "strong", "oud", "spicy"])
                        )
                    ]
                    if bolder_candidates:
                        buttons.append({"label": "Show Bolder Option", "action": "show_bolder", "type": "secondary"})
                    
                    # Add "other options" button if more products available
                    if remaining > 0:
                        buttons.append({"label": "Other Options", "action": "show_other", "type": "primary"})
                        reply_parts.append("Want to know more or see other options?")
                    else:
                        reply_parts.append("Want to know more?")
                    
                    # Add "View Product" button
                    product_url = top.get("product_url", "")
                    if product_url:
                        buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
                    
                    if buttons:
                        state["button_suggestions"] = buttons
                    
                    reply = " ".join(reply_parts)
                    
                    state["messages"].append({"role": "assistant", "content": reply})
                    logger.info("node=response used_recommendations=1 showing_product_index=%d remaining=%d buttons=%d", 
                               current_index, remaining, len(buttons))
                    return state
                else:
                    # No more recommendations
                    total_shown = len(recs)
                    if total_shown == 1:
                        reply = "That's the only option matching your preferences. Would you like to adjust your requirements?"
                    else:
                        reply = f"Those are the {total_shown} options matching your preferences. Would you like to adjust your requirements to see more?"
                    state["messages"].append({"role": "assistant", "content": reply})
                    logger.info("node=response exhausted_recommendations total_shown=%d", total_shown)
                    return state
            else:
                # Sufficient preferences but NO recommendations found (e.g. strict budget filter)
                # Check what might be the blocker
                buttons = []
                reply = "I couldn't find any perfumes matching your exact criteria."
                
                if has_budget:
                    # Offer budget adjustment options
                    reply += " It might be the budget constraint. Would you like to see options with a slightly higher budget?"
                    
                    # Set awaiting_choice so collect_preferences knows user is adjusting budget
                    state["awaiting_choice"] = "budget"
                    
                    # Get current budget
                    budget_info = prefs.get("budget", {})
                    if isinstance(budget_info, dict):
                        current_amount = budget_info.get("amount", 0)
                        operator = budget_info.get("operator", "under")
                        
                        # Offer higher budget options
                        if operator == "under":
                            if current_amount < 700:
                                buttons.append({"label": "Under ₹700", "action": "select_budget:under_700", "type": "primary"})
                            if current_amount < 900:
                                buttons.append({"label": "₹700-₹900", "action": "select_budget:700_900", "type": "primary"})
                            buttons.append({"label": "No Budget Limit", "action": "select_budget:no_limit", "type": "primary"})
                        
                        # Also offer to change other preferences
                        buttons.append({"label": "Change Vibe", "action": "change_vibe", "type": "secondary"})
                    else:
                        # Fallback buttons
                        buttons = [
                            {"label": "Under ₹700", "action": "select_budget:under_700", "type": "primary"},
                            {"label": "₹700-₹900", "action": "select_budget:700_900", "type": "primary"},
                            {"label": "No Budget Limit", "action": "select_budget:no_limit", "type": "primary"},
                        ]
                else:
                    # Offer to adjust other preferences
                    reply += " Would you like to adjust your preferences?"
                    buttons = [
                        {"label": "Change Gender", "action": "change_gender", "type": "primary"},
                        {"label": "Change Vibe", "action": "change_vibe", "type": "primary"},
                    ]
                    if not is_gift:
                        buttons.append({"label": "Change Occasion", "action": "change_occasion", "type": "primary"})
                
                if buttons:
                    state["button_suggestions"] = buttons
                
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response no_recommendations_found sufficient_prefs=1 buttons=%d", len(buttons))
                return state
        
        # Ask for missing preferences ONE AT A TIME with button suggestions
        # Priority: gender > vibe/scent_type > occasion > budget
        buttons = []
        
        # Use saved awaiting_choice for preference detection
        current_awaiting = awaiting_choice_before_clear
        
        if is_gift:
            # Gift flow: ask gender -> vibe (personality) -> budget
            if not has_gender:
                reply = "Who is this gift for?"
                buttons = [
                    {"label": "For Him", "action": "select_gender:for_him", "type": "primary"},
                    {"label": "For Her", "action": "select_gender:for_her", "type": "primary"},
                    {"label": "Unisex", "action": "select_gender:unisex", "type": "secondary"},
                ]
                state["awaiting_choice"] = "gender"
                if "gender" not in questions_asked:
                    questions_asked.append("gender")
                    state["questions_asked"] = questions_asked
            elif "vibe" not in questions_asked:
                # Have gender, ask about recipient's personality (maps to vibe/scent)
                # Use appropriate pronouns based on gift recipient gender
                gender = prefs.get("gender")
                if gender == "for_him":
                    reply = "What's his personality like?"
                elif gender == "for_her":
                    reply = "What's her personality like?"
                else:
                    reply = "What's their personality like?"
                
                buttons = [
                    {"label": "Classy & Elegant", "action": "select_vibe:floral", "type": "primary"},
                    {"label": "Playful & Fun", "action": "select_vibe:fresh", "type": "primary"},
                    {"label": "Sweet & Romantic", "action": "select_vibe:sweet", "type": "primary"},
                    {"label": "Intense & Bold", "action": "select_vibe:spicy", "type": "primary"},
                ]
                state["awaiting_choice"] = "vibe"
                questions_asked.append("vibe")
                state["questions_asked"] = questions_asked
            elif "budget" not in questions_asked:
                # Have gender and vibe, ask for budget
                reply = "What's your budget range?"
                buttons = [
                    {"label": "Under ₹700", "action": "select_budget:under_700", "type": "primary"},
                    {"label": "₹700-₹900", "action": "select_budget:700_900", "type": "primary"},
                    {"label": "No Limit", "action": "select_budget:no_limit", "type": "secondary"},
                ]
                state["awaiting_choice"] = "budget"
                questions_asked.append("budget")
                state["questions_asked"] = questions_asked
            else:
                # All gift questions asked, make recommendation
                reply = "Let me find the perfect gift set for you."
        else:
            # Regular product recommendation flow - ask ONE question at a time
            if not has_gender:
                reply = "Who is this for?"
                buttons = [
                    {"label": "For Him", "action": "select_gender:for_him", "type": "primary"},
                    {"label": "For Her", "action": "select_gender:for_her", "type": "primary"},
                    {"label": "Unisex", "action": "select_gender:unisex", "type": "secondary"},
                ]
                state["awaiting_choice"] = "gender"
                if "gender" not in questions_asked:
                    questions_asked.append("gender")
                    state["questions_asked"] = questions_asked
            elif not has_scent_type:
                # Have gender, ask for vibe/scent type next
                reply = "What kind of vibe are you looking for?"
                buttons = [
                    {"label": "Fresh & Breezy", "action": "select_vibe:fresh", "type": "primary"},
                    {"label": "Sweet & Gourmand", "action": "select_vibe:sweet", "type": "primary"},
                    {"label": "Spicy & Bold", "action": "select_vibe:spicy", "type": "primary"},
                    {"label": "Oud & Intense", "action": "select_vibe:oud", "type": "primary"},
                    {"label": "Soft & Cozy", "action": "select_vibe:floral", "type": "secondary"},
                    {"label": "Woody & Earthy", "action": "select_vibe:woody", "type": "secondary"},
                ]
                state["awaiting_choice"] = "vibe"
                if "vibe" not in questions_asked:
                    questions_asked.append("vibe")
                    state["questions_asked"] = questions_asked
            elif "occasion" not in questions_asked:
                # Have gender and scent, ask for occasion
                reply = "When will you wear it?"
                buttons = [
                    {"label": "Daily", "action": "select_occasion:casual", "type": "primary"},
                    {"label": "Office", "action": "select_occasion:formal", "type": "primary"},
                    {"label": "Date Night", "action": "select_occasion:date_night", "type": "primary"},
                    {"label": "Party", "action": "select_occasion:party", "type": "primary"},
                    {"label": "Mixed", "action": "select_occasion:mixed", "type": "secondary"},
                ]
                state["awaiting_choice"] = "occasion"
                questions_asked.append("occasion")
                state["questions_asked"] = questions_asked
            elif "budget" not in questions_asked:
                # Have all other prefs, ask for budget
                reply = "What's your budget range?"
                buttons = [
                    {"label": "Under ₹700", "action": "select_budget:under_700", "type": "primary"},
                    {"label": "₹700-₹900", "action": "select_budget:700_900", "type": "primary"},
                    {"label": "No Limit", "action": "select_budget:no_limit", "type": "secondary"},
                ]
                state["awaiting_choice"] = "budget"
                questions_asked.append("budget")
                state["questions_asked"] = questions_asked
            else:
                # All questions have been asked, make recommendation
                reply = "Let me find the perfect perfume for you."
        
        if buttons:
            state["button_suggestions"] = buttons
        
        state["messages"].append({"role": "assistant", "content": reply})
        logger.info("node=response ask_clarifying_for_reco=1 missing_gender=%s missing_scent=%s missing_budget=%s is_gift=%s buttons=%d", 
                   not has_gender, not has_scent_type, not has_budget, is_gift, len(buttons))
        return state

    # If product_info but no exact match, check if user is asking about previously recommended products
    if intent == "product_info":
        last_recs = state.get("last_recommendations", [])
        if last_recs:
            # User might be asking about a previously recommended product
            # Check if the question matches product_question patterns
            user_text_lower = user_text.lower()
            if any(phrase in user_text_lower for phrase in ["what", "price", "cost", "contents", "includes", "what's in", "what do i get"]):
                # Treat as product_question instead
                product = last_recs[0] if last_recs else {}
                price = product.get("price", 0)
                name = product.get("name", "this product")
                desc = product.get("short_description", "")
                notes = product.get("notes", {})
                
                if "price" in user_text_lower or "cost" in user_text_lower:
                    if price:
                        reply = f"The price of {name} is ₹{price}. Can I help you with anything else?"
                    else:
                        reply = f"I don't have the price information for {name} right now. Can I help you with anything else?"
                elif any(phrase in user_text_lower for phrase in ["what do i get", "what's in", "what is in", "contents", "includes"]):
                    reply_parts = [f"{name}"]
                    if desc:
                        reply_parts.append(f"includes: {desc}")
                    all_notes = []
                    if notes:
                        all_notes = (
                            notes.get("head", []) + 
                            notes.get("heart", []) + 
                            notes.get("base", [])
                        )
                    if all_notes:
                        notes_str = ", ".join(all_notes[:5])
                        reply_parts.append(f"Key notes: {notes_str}")
                    reply = ". ".join(reply_parts) + ". Can I help you with anything else?"
                else:
                    # Generic product info question
                    reply_parts = [f"{name}"]
                    if desc:
                        reply_parts.append(desc)
                    if price:
                        reply_parts.append(f"Priced at ₹{price}.")
                    reply = ". ".join(reply_parts) + " Can I help you with anything else?"
                
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response answered_product_info_with_context=1")
                return state
        
        # No last recommendations - proceed with normal product_info flow
        if isinstance(tool_result, dict) and tool_result.get("products") and not tool_result.get("product"):
            reply = (
                "We have a few options. To narrow it down, what gender are you looking for (men's, women's, or unisex), "
                "what scent type do you prefer, and what's your budget? "
                "I can recommend one that fits best. Can I help you with anything else?"
            )
            state["messages"].append({"role": "assistant", "content": reply})
            logger.info("node=response ask_clarifying_for_product_info=1")
            return state

    # Handle fallback intent - offer human handoff if bot doesn't understand
    if intent == "fallback":
        reply = "I'm not sure I understand. Would you like to talk with a human agent who can help you better?"
        buttons = [
            {"label": "Yes, connect me", "action": "confirm_handoff", "type": "primary"},
            {"label": "No, thanks", "action": "cancel_handoff", "type": "secondary"},
        ]
        state["button_suggestions"] = buttons
        state["handoff_state"] = "awaiting_confirmation"
        state["messages"].append({"role": "assistant", "content": reply})
        logger.info("node=response fallback_intent offering_handoff=1")
        return state

    tool_context = ""
    if tool_result:
        tool_context = f"Tool result: {tool_result}"

    final_prompt = f"""{ASSISTANT_STYLE_PROMPT}

User said: {user_text}
Detected intent: {intent}
{tool_context}

Write a friendly, human, short answer. Do not use Markdown or lists. Keep it conversational.
End with: "Can I help you with anything else?"
"""
    answer = _call_llm(final_prompt)
    logger.info("node=response llm_called=1 answer_len=%d", len(answer) if isinstance(answer, str) else -1)

    state["messages"].append({"role": "assistant", "content": answer})
    return state


graph = StateGraph(ChatState)
graph.set_entry_point("intent_router")
graph.add_node("intent_router", intent_router)
graph.add_node("auth_guard", auth_guard)
graph.add_node("collect_preferences", collect_preferences)
graph.add_node("tool_exec", tool_exec)
graph.add_node("response_node", response_node)

graph.add_edge("intent_router", "auth_guard")
graph.add_edge("auth_guard", "collect_preferences")
graph.add_edge("collect_preferences", "tool_exec")
graph.add_edge("tool_exec", "response_node")
graph.add_edge("response_node", END)

compiled_app = graph.compile()


import os
from typing import TypedDict, List, Optional, Dict, Any
import re
import logging
import time

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from openai import OpenAI

from prompts import INTENT_PROMPT, ASSISTANT_STYLE_PROMPT, PREFERENCE_EXTRACTION_PROMPT
from tools import (
    get_product_by_name,
    list_products,
    recommend_products,
    get_order_status,
    get_return_policy,
    get_byob_products,
    validate_byob_selection,
    get_faq_answer,
)

try:
    from recommender import extract_preferences_and_recommend
except Exception:
    extract_preferences_and_recommend = None

logger = logging.getLogger("blabli.graph")
_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger.setLevel(_level)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
    logger.addHandler(_h)
logger.propagate = False


def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Put it in .env at project root.")
    return OpenAI(api_key=api_key)


class ChatState(TypedDict, total=False):
    messages: List[dict]
    intent: Optional[str]
    customer_authed: bool
    customer_email: Optional[str]
    order_number: Optional[str]
    tool_result: Optional[dict]
    missing_field: Optional[str]
    preferences: Optional[Dict[str, Any]]
    recommendations: Optional[List[Dict[str, Any]]]
    last_recommendations: Optional[List[Dict[str, Any]]]  # Track last recommended products for follow-up questions
    recommendation_index: int  # Track which recommendation we've shown (0-indexed)
    byob_selections: Optional[List[str]]  # Track BYOB product selections (IDs)
    byob_max_items: int  # Max items allowed in BYOB (default 3)
    button_suggestions: Optional[List[Dict[str, str]]]  # Button suggestions to show to user
    awaiting_choice: Optional[str]  # Track what choice we're waiting for (vibe, occasion, gender, etc.)
    questions_asked: Optional[List[str]]  # Track which questions have been asked (gender, vibe, occasion, budget)
    handoff_state: Optional[str]  # Track human handoff flow: None, "awaiting_confirmation", "collecting_email", "collecting_phone", "completed"
    handoff_email: Optional[str]  # Email collected for handoff
    handoff_phone: Optional[str]  # Phone collected for handoff
    handoff_completed: bool  # Flag to stop bot after handoff


def _call_llm(prompt: str) -> str:
    """
    Wrapper that works with both the modern OpenAI Responses API (>=1.0)
    and the legacy Chat Completions API (<1.0).
    """
    client = get_client()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if hasattr(client, "responses"):
        resp = client.responses.create(model=model, input=prompt)
        return resp.output[0].content[0].text.strip()

    # fallback for older openai package versions
    if not hasattr(client, "chat") or not hasattr(client.chat, "completions"):
        raise RuntimeError("OpenAI client does not support responses or chat completions APIs.")

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    message = resp.choices[0].message
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    return content.strip()


def _call_intent_llm(prompt: str) -> str:
    # Prompt is now constructed by the caller (intent_router) to include context
    return _call_llm(prompt)


def intent_router(state: ChatState):
    logger.info("node=intent_router entering")
    last_user_msg = ""
    last_assistant_msg = ""
    
    # Find last user message and last assistant message
    for m in reversed(state["messages"]):
        if m["role"] == "user" and not last_user_msg:
            last_user_msg = m["content"]
        elif m["role"] == "assistant" and not last_assistant_msg:
            last_assistant_msg = m["content"]
        
        if last_user_msg and last_assistant_msg:
            break
    
    # Direct intent mapping for button actions (bypassing LLM for efficiency)
    user_msg_lower = last_user_msg.lower()
    
    # Handle handoff button actions
    if "confirm_handoff" in last_user_msg or user_msg_lower in ["yes", "yeah", "yep", "sure", "ok", "okay", "connect me"]:
        handoff_state = state.get("handoff_state")
        if handoff_state == "awaiting_confirmation":
            state["intent"] = "human_handoff"
            logger.info("node=intent_router handoff_confirmed=1")
            return state
    
    # Handle small talk after handoff completion or other completed actions
    # Check if user is just acknowledging/thanking after we completed something
    gratitude_phrases = ["thank you", "thanks", "thank", "ok", "okay", "great", "perfect", "awesome", "good", "appreciate"]
    is_short_gratitude = (
        len(last_user_msg.split()) <= 3 and  # Short message
        any(phrase in user_msg_lower for phrase in gratitude_phrases) and
        not any(q in user_msg_lower for q in ["?", "what", "where", "when", "how", "why", "can", "could", "would"])  # Not a question
    )
    
    # Check if last assistant message was a completion message (ticket creation, recommendation shown, etc.)
    completion_indicators = ["ticket id:", "can i help you with anything else", "someone will contact you"]
    last_was_completion = any(indicator in last_assistant_msg.lower() for indicator in completion_indicators) if last_assistant_msg else False
    
    if is_short_gratitude and last_was_completion:
        state["intent"] = "small_talk"
        logger.info("node=intent_router detected_post_completion_gratitude=1 routing_to_small_talk")
        return state
    
    # Special handling: If user is answering a direct choice question with a button-like response
    # Only continue intent if they're literally answering the question we asked
    awaiting_choice = state.get("awaiting_choice")
    if awaiting_choice:
        last_intent = state.get("intent")
        # Check if this is a button-style answer to the question we asked
        is_button_answer = False
        
        if awaiting_choice == "byob_selection":
            # If we're awaiting BYOB selection, preserve build_own_box intent
            # User might be clicking a product button or typing a product name
            if last_intent == "build_own_box":
                logger.info("node=intent_router continuing_intent=build_own_box awaiting=byob_selection")
                state["intent"] = "build_own_box"
                return state
        elif awaiting_choice == "gender" and any(word in user_msg_lower for word in ["for him", "for her", "unisex", "men's", "women's"]):
            is_button_answer = True
        elif awaiting_choice == "vibe" and any(word in user_msg_lower for word in ["fresh", "sweet", "spicy", "oud", "floral", "woody", "breezy", "gourmand", "cozy", "earthy"]):
            is_button_answer = True
        elif awaiting_choice == "occasion" and any(word in user_msg_lower for word in ["daily", "office", "date night", "party", "mixed", "casual", "formal"]):
            is_button_answer = True
        elif awaiting_choice == "budget" and any(word in user_msg_lower for word in ["under", "700", "900", "no limit", "any", "₹"]):
            is_button_answer = True
        
        if is_button_answer and last_intent in ("find_perfume", "product_recommendation", "gift_recommendation"):
            logger.info("node=intent_router continuing_intent=%s awaiting=%s detected_button_answer=1", last_intent, awaiting_choice)
            state["intent"] = last_intent
            return state
    
    direct_intent_map = {
        "find my perfume": "find_perfume",
        "gift for someone": "gift_recommendation",
        "build my own box": "build_own_box",
        "track my order": "order_status",
        "looking for perfume": "find_perfume",
        "want a gift": "gift_recommendation",
        "create my box": "build_own_box",
        "track order": "order_status",
        "talk to human": "human_handoff",
        "speak with human": "human_handoff",
        "talk to agent": "human_handoff",
        "speak with agent": "human_handoff",
        "talk to representative": "human_handoff",
        "speak with representative": "human_handoff",
        "talk to support": "human_handoff",
        "speak with support": "human_handoff",
        "human agent": "human_handoff",
        "connect me to human": "human_handoff",
    }
    
    for phrase, mapped_intent in direct_intent_map.items():
        if phrase in user_msg_lower:
            state["intent"] = mapped_intent
            logger.info("node=intent_router direct_map_intent=%s", mapped_intent)
            return state
    
    # Check if this is a product question about the last recommended product
    has_last_recs = bool(state.get("last_recommendations"))
    if has_last_recs:
        # Common product question patterns
        product_question_patterns = [
            "what is the price", "what is its price", "how much", "what's the price",
            "what do i get", "what's in it", "what comes with", "contents",
            "when can i wear", "when to wear", "when should i wear", "suitable for",
            "what occasion", "what notes", "tell me more", "more about",
        ]
        if any(pattern in user_msg_lower for pattern in product_question_patterns):
            state["intent"] = "product_question"
            logger.info("node=intent_router direct_map_product_question=1")
            return state
    
    # Check if user is asking about previously recommended products
    # This helps improve intent detection for product_question
    
    # Construct prompt with context
    prompt = INTENT_PROMPT
    if last_assistant_msg:
        prompt += f"\nAssistant asked: {last_assistant_msg}"
    
    if has_last_recs:
        # Add context to help intent detection
        prompt += f"\nContext: Products were just recommended."
        
    prompt += f"\nUser: {last_user_msg}"
    
    intent = _call_intent_llm(prompt)
    
    state["intent"] = intent
    logger.info("node=intent_router detected_intent=%s has_last_recs=%s", intent, has_last_recs)
    return state


def auth_guard(state: ChatState):
    logger.info(
        "node=auth_guard intent=%s email_present=%s order_present=%s",
        state.get("intent"),
        bool(state.get("customer_email")),
        bool(state.get("order_number")),
    )
    # If we're in an auth-requiring flow, try to extract missing fields from latest user message
    requires_auth = state["intent"] in ("order_status", "change_address") or state.get("missing_field") == "auth"

    if requires_auth:
        last_user_msg = ""
        for m in reversed(state["messages"]):
            if m["role"] == "user":
                last_user_msg = m["content"]
                break

        # Fill email if missing
        if not state.get("customer_email") and last_user_msg:
            email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", last_user_msg)
            if email_match:
                state["customer_email"] = email_match.group(0)
                logger.info("node=auth_guard extracted_email=1")

        # Fill order number if missing (e.g., BLB-1001)
        if not state.get("order_number") and last_user_msg:
            order_match = re.search(r"\bBLB-\d{3,}\b", last_user_msg, flags=re.IGNORECASE)
            if order_match:
                state["order_number"] = order_match.group(0).upper()
                logger.info("node=auth_guard extracted_order_number=1")

        if not state.get("customer_email") or not state.get("order_number"):
            state["missing_field"] = "auth"
            state["customer_authed"] = False
            logger.info("node=auth_guard auth_status=missing")
        else:
            state["missing_field"] = None
            state["customer_authed"] = True
            logger.info("node=auth_guard auth_status=ok")
    else:
        state["missing_field"] = None
    return state


def collect_preferences(state: ChatState):
    """
    For product-related intents, normalize or initialize the user's preference profile
    before we hit tools / LLM. This makes the bot contextual across turns.
    """
    intent = state.get("intent")
    if intent not in ("product_info", "product_recommendation", "find_perfume", "gift_recommendation"):
        return state

    # Get all user messages from conversation history to extract preferences from context
    user_messages = [m["content"] for m in state["messages"] if m["role"] == "user"]
    last_user_msg = user_messages[-1] if user_messages else ""
    last_user_msg_lower = last_user_msg.lower()
    # Combine recent user messages for better context (last 3 messages)
    context_text = " ".join(user_messages[-3:]) if len(user_messages) > 0 else ""

    # ensure preferences dict exists (adapted for perfumes) - preserve existing preferences
    prefs = state.get("preferences") or {
        "gender": None,
        "scent_types": [],
        "budget": None,
        "occasions": [],
        "vibes": [],
        "specific_notes": [],
        "is_gift": False,
        "occasion_raw": None,  # Store raw user input for occasion (e.g., "wedding", "date night")
    }
    
    # Direct preference detection from button-like responses (faster than LLM)
    # This handles cases where user types instead of clicking buttons
    awaiting_choice = state.get("awaiting_choice")
    
    if awaiting_choice == "gender" and not prefs.get("gender"):
        # User was asked about gender, detect their answer
        # Check for specific gender terms first (more specific patterns win)
        if any(word in last_user_msg_lower for word in ["men's", "mens", "for him", "male perfume", "man's", "guy's"]):
            prefs["gender"] = "for_him"
            logger.info("node=collect_preferences direct_detect_gender=for_him")
        elif any(word in last_user_msg_lower for word in ["women's", "womens", "for her", "female perfume", "woman's", "lady's"]):
            prefs["gender"] = "for_her"
            logger.info("node=collect_preferences direct_detect_gender=for_her")
        elif "unisex" in last_user_msg_lower or "both" in last_user_msg_lower or "either" in last_user_msg_lower:
            prefs["gender"] = "unisex"
            logger.info("node=collect_preferences direct_detect_gender=unisex")
        # If they just say generic words without specific gender indicator, try to detect
        elif any(word in last_user_msg_lower for word in ["men", "male", "him", "guy", "man"]) and "women" not in last_user_msg_lower:
            prefs["gender"] = "for_him"
            logger.info("node=collect_preferences direct_detect_gender=for_him (generic)")
        elif any(word in last_user_msg_lower for word in ["women", "female", "her", "lady", "woman"]) and "men" not in last_user_msg_lower:
            prefs["gender"] = "for_her"
            logger.info("node=collect_preferences direct_detect_gender=for_her (generic)")
    
    if awaiting_choice == "vibe" and not prefs.get("scent_types"):
        # User was asked about vibe/scent
        vibe_map = {
            "fresh": ["fresh", "breezy", "citrus", "clean"],
            "sweet": ["sweet", "gourmand", "vanilla", "honey"],
            "spicy": ["spicy", "bold", "warm"],
            "oud": ["oud", "intense", "habibi"],
            "floral": ["floral", "soft", "cozy", "rose"],
            "woody": ["woody", "earthy", "wood"],
        }
        for vibe_type, keywords in vibe_map.items():
            if any(kw in last_user_msg_lower for kw in keywords):
                prefs["scent_types"] = [vibe_type]
                logger.info("node=collect_preferences direct_detect_vibe=%s", vibe_type)
                break
    
    if awaiting_choice == "occasion" and not prefs.get("occasions"):
        # User was asked about occasion
        occasion_map = {
            "casual": ["daily", "everyday", "casual"],
            "formal": ["office", "work", "formal", "business", "wedding", "ceremony", "function", "event"],
            "date_night": ["date", "romantic", "date night"],
            "party": ["party", "night", "clubbing"],
            "mixed": ["mixed", "all", "everything", "any"],
        }
        for occ_type, keywords in occasion_map.items():
            if any(kw in last_user_msg_lower for kw in keywords):
                prefs["occasions"] = [occ_type]
                # Store raw occasion text for natural reference later
                for keyword in keywords:
                    if keyword in last_user_msg_lower:
                        prefs["occasion_raw"] = keyword
                        break
                logger.info("node=collect_preferences direct_detect_occasion=%s raw=%s", occ_type, prefs.get("occasion_raw"))
                break
    
    # Handle budget updates (both when awaiting and when user spontaneously changes)
    if awaiting_choice == "budget" or any(word in last_user_msg_lower for word in ["budget", "above", "under", "over", "₹"]):
        # User was asked about budget or is updating budget preference
        # Allow updates even if budget is already set (user can change their mind)
        budget_updated = False
        
        if "under" in last_user_msg_lower and "700" in last_user_msg_lower:
            prefs["budget"] = {"amount": 700, "operator": "under"}
            logger.info("node=collect_preferences direct_detect_budget=under_700")
            budget_updated = True
        elif ("above" in last_user_msg_lower or "over" in last_user_msg_lower or "more than" in last_user_msg_lower):
            # Extract amount dynamically
            import re
            amount_match = re.search(r'(\d+)', last_user_msg_lower)
            if amount_match:
                amount = int(amount_match.group(1))
                prefs["budget"] = {"amount": amount, "operator": "over"}
                logger.info("node=collect_preferences direct_detect_budget=above_%d", amount)
                budget_updated = True
        elif "700" in last_user_msg_lower and "900" in last_user_msg_lower:
            prefs["budget"] = {"amount": 800, "operator": "around"}
            logger.info("node=collect_preferences direct_detect_budget=700_900")
            budget_updated = True
        elif "no limit" in last_user_msg_lower or ("any" in last_user_msg_lower and ("budget" in last_user_msg_lower or awaiting_choice == "budget")):
            prefs["budget"] = None  # No budget constraint
            logger.info("node=collect_preferences direct_detect_budget=no_limit")
            budget_updated = True
        
        # Mark budget as asked if it was updated
        if budget_updated and "budget" not in state.get("questions_asked", []):
            questions_asked = state.get("questions_asked", [])
            questions_asked.append("budget")
            state["questions_asked"] = questions_asked

    # Use LLM to extract preferences from context (Smart extraction)
    try:
        llm_prompt = f"""{PREFERENCE_EXTRACTION_PROMPT}

Conversation Context:
{context_text}
"""
        llm_response = _call_llm(llm_prompt)
        # Clean up response to ensure it's valid JSON
        llm_response = llm_response.strip()
        if llm_response.startswith("```json"):
            llm_response = llm_response[7:]
        if llm_response.endswith("```"):
            llm_response = llm_response[:-3]
        
        import json
        extracted_prefs = json.loads(llm_response)
        
        # Merge extracted prefs into existing prefs (only update if not already set or explicitly changed)
        if extracted_prefs.get("gender"):
            # Only update gender if not already set
            if not prefs.get("gender"):
                prefs["gender"] = extracted_prefs["gender"]
        
        if extracted_prefs.get("scent_types"):
            existing_scents = prefs.get("scent_types", [])
            new_scents = extracted_prefs["scent_types"]
            # Only add new scents, don't replace existing ones
            prefs["scent_types"] = list(set(existing_scents + new_scents))
            
        if extracted_prefs.get("budget") and not budget_updated:
            # Only update budget from LLM if it wasn't already updated by direct detection above
            prefs["budget"] = extracted_prefs["budget"]
            
        if extracted_prefs.get("is_gift") is not None:
            # Only update is_gift if explicitly detected AND not already set
            # Preserve existing is_gift value unless user explicitly says it's (not) a gift
            if extracted_prefs["is_gift"] and not prefs.get("is_gift"):
                # User explicitly mentioned it's a gift, update it
                prefs["is_gift"] = True
            # Don't change is_gift to False if it was already True
            # User would need to explicitly say "not a gift" to change it

        if extracted_prefs.get("specific_notes"):
            existing_notes = prefs.get("specific_notes", [])
            new_notes = extracted_prefs["specific_notes"]
            prefs["specific_notes"] = list(set(existing_notes + new_notes))
            
        logger.info("node=collect_preferences llm_extraction_success=1 extracted=%s", extracted_prefs)
    except Exception as e:
        logger.error("node=collect_preferences llm_extraction_error=%s", e)

    # If we have a recommender module, let it do smarter extraction (Rule-based + Scoring)
    # Use context_text to extract preferences from entire conversation, not just last message
    if extract_preferences_and_recommend is not None:
        try:
            # Get recently recommended IDs to avoid repetition
            last_recs = state.get("last_recommendations") or []
            recent_ids = [p.get("id") for p in last_recs if isinstance(p, dict) and "id" in p]
            
            # Pass the updated prefs (with LLM extraction) to the recommender
            # Pass context_text for extraction, but last_user_msg for boosting
            rec_out = extract_preferences_and_recommend(
                context_text, 
                prefs, 
                latest_user_text=last_user_msg,
                recently_recommended_ids=recent_ids
            )
            # Support both tuple-style and dict-style return values
            if isinstance(rec_out, tuple):
                # recommender returned (new_prefs, recos)
                new_prefs, recos = rec_out
            else:
                # recommender returned {"preferences": {...}, "recommendations": [...]}
                new_prefs = rec_out.get("preferences") or prefs
                recos = rec_out.get("recommendations") or []
            # Update state
            state["preferences"] = new_prefs or prefs
            
            # Only set recommendations if we have sufficient preferences
            # For gift queries, need gender + vibe + budget (3 questions)
            # For regular queries, need ALL 4 questions asked
            is_gift = new_prefs.get("is_gift", False)
            questions_asked = state.get("questions_asked", [])
            required_questions = {"gender", "vibe", "occasion", "budget"}
            all_questions_asked = required_questions.issubset(set(questions_asked))
            
            if is_gift:
                gift_required_questions = {"gender", "vibe", "budget"}
                has_sufficient_prefs = gift_required_questions.issubset(set(questions_asked))
            else:
                has_sufficient_prefs = all_questions_asked
            
            if recos and has_sufficient_prefs:
                # pre-fill tool_result so tool_exec/response_node can use it directly
                state["tool_result"] = {"recommendations": recos}
            else:
                # Clear any pre-existing recommendations if we don't have enough preferences
                state["tool_result"] = None
            return state
        except Exception as e:
            logger.exception("node=collect_preferences error=%s", e)

    state["preferences"] = prefs
    return state


def tool_exec(state: ChatState):
    logger.info("node=tool_exec intent=%s", state.get("intent"))
    # if a previous node already put recommendations into tool_result, keep it
    if state.get("tool_result") and "recommendations" in state["tool_result"]:
        logger.info("node=tool_exec detected_pre_filled_recommendations=1")
        # Reset recommendation index when new recommendations are generated
        state["recommendation_index"] = 0
        return state

    intent = state["intent"]
    result = {}

    if state.get("missing_field") == "auth":
        state["tool_result"] = None
        logger.info("node=tool_exec blocked_missing_auth=1")
        return state

    if intent == "product_info":
        user_text = next(m["content"] for m in reversed(state["messages"]) if m["role"] == "user")
        prod = get_product_by_name(user_text)
        if prod:
            result = {"product": prod}
            logger.info("node=tool_exec path=product_info hit=exact")
        else:
            result = {"products": list_products()}
            logger.info("node=tool_exec path=product_info hit=list")
    elif intent in ("product_recommendation", "find_perfume", "gift_recommendation"):
        # If collect_preferences already set recommendations (with sufficient prefs), use them
        if state.get("tool_result") and isinstance(state.get("tool_result"), dict) and "recommendations" in state["tool_result"]:
            logger.info("node=tool_exec path=%s using_pre_filled_recommendations", intent)
            # Reset recommendation index when new recommendations are generated
            state["recommendation_index"] = 0
            return state
        
        # If collect_preferences set tool_result to None (insufficient prefs), preserve it
        if state.get("tool_result") is None:
            logger.info("node=tool_exec path=%s preserving_none_from_collect_prefs", intent)
            return state
        
        # Check if we have sufficient preferences before recommending
        prefs = state.get("preferences") or {}
        has_gender = prefs.get("gender") is not None
        has_scent_type = isinstance(prefs.get("scent_types"), list) and len(prefs.get("scent_types", [])) > 0
        has_budget = prefs.get("budget") is not None
        has_occasion = isinstance(prefs.get("occasions"), list) and len(prefs.get("occasions", [])) > 0
        is_gift = prefs.get("is_gift", False) or intent == "gift_recommendation"
        
        # Check if all required questions have been asked
        questions_asked = state.get("questions_asked", [])
        required_questions = {"gender", "vibe", "occasion", "budget"}
        all_questions_asked = required_questions.issubset(set(questions_asked))
        
        # For gift queries, need gender + vibe + budget (3 questions)
        # For regular queries, need ALL 4 questions asked
        if is_gift:
            gift_required_questions = {"gender", "vibe", "budget"}
            has_sufficient_prefs = gift_required_questions.issubset(set(questions_asked))
        else:
            has_sufficient_prefs = all_questions_asked
        
        # Only recommend if we have sufficient preferences
        if has_sufficient_prefs:
            user_text = next(m["content"] for m in reversed(state["messages"]) if m["role"] == "user")
            recs = recommend_products(user_text)
            result = {"recommendations": recs}
            # Reset recommendation index when new recommendations are generated
            state["recommendation_index"] = 0
            logger.info("node=tool_exec path=%s count=%d", intent, len(recs))
        else:
            # No sufficient preferences - don't set recommendations, response_node will ask questions
            result = None
            logger.info("node=tool_exec path=%s insufficient_prefs gender=%s scent=%s budget=%s is_gift=%s", 
                       intent, has_gender, has_scent_type, has_budget, is_gift)
    elif intent == "build_own_box":
        # Get BYOB products
        byob_products = get_byob_products()
        result = {"byob_products": byob_products}
        logger.info("node=tool_exec path=build_own_box count=%d", len(byob_products))
    elif intent == "general_question":
        user_text = next(m["content"] for m in reversed(state["messages"]) if m["role"] == "user")
        faq_answer = get_faq_answer(user_text)
        result = {"faq_answer": faq_answer}
        logger.info("node=tool_exec path=general_question found=%s", bool(faq_answer))
    elif intent == "returns_refunds":
        result = get_return_policy()
        logger.info("node=tool_exec path=returns_refunds")
    elif intent == "order_status":
        r = get_order_status(state["customer_email"], state["order_number"])
        result = r
        logger.info("node=tool_exec path=order_status error=%s", str(r.get("error")) if isinstance(r, dict) else "none")
    elif intent == "change_address":
        result = {"ok": False, "message": "Address change not allowed after dispatch."}
        logger.info("node=tool_exec path=change_address")
    else:
        result = {}
        logger.info("node=tool_exec path=unknown")

    state["tool_result"] = result
    return state


def response_node(state: ChatState):
    logger.info("node=response entering intent=%s missing=%s", state.get("intent"), state.get("missing_field"))
    user_text = next(m["content"] for m in reversed(state["messages"]) if m["role"] == "user")
    intent = state.get("intent")
    tool_result = state.get("tool_result")
    missing = state.get("missing_field")
    customer_authed = state.get("customer_authed", False)
    customer_email = state.get("customer_email")
    order_number = state.get("order_number")
    
    # Save awaiting_choice before clearing (needed for BYOB and preference flows)
    awaiting_choice_before_clear = state.get("awaiting_choice")
    
    # Clear button suggestions at start of response (will be regenerated if needed)
    state["button_suggestions"] = None
    # Don't clear awaiting_choice yet - we need it for BYOB and preference detection
    # It will be cleared or updated later in the flow

    # Handle human handoff flow
    handoff_state = state.get("handoff_state")
    if intent == "human_handoff" or handoff_state:
        # Force intent to human_handoff if we're in handoff flow
        if handoff_state:
            state["intent"] = "human_handoff"
        
        # Extract email/phone from user message if provided
        user_text_lower = user_text.lower()
        email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", user_text)
        phone_match = re.search(r"[\d\s\-\+\(\)]{10,}", user_text)
        
        if handoff_state == "collecting_email":
            # User is providing email
            if email_match:
                state["handoff_email"] = email_match.group(0)
                state["handoff_state"] = "collecting_phone"
                reply = "Thank you! Please provide your phone number so we can contact you."
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response handoff_email_collected=%s", state["handoff_email"])
                return state
            else:
                reply = "I didn't catch a valid email address. Please provide your email address."
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response handoff_email_invalid=1")
                return state
        elif handoff_state == "collecting_phone":
            # User is providing phone
            if phone_match:
                state["handoff_phone"] = phone_match.group(0).strip()
                # Generate ticket (in real implementation, this would call a ticket system API)
                ticket_id = f"TKT-{int(time.time())}"
                state["handoff_state"] = None  # Clear handoff state to allow normal conversation
                state["handoff_completed"] = True  # Keep flag for reference but don't block
                reply = f"Ok, we have raised the ticket (Ticket ID: {ticket_id}). Someone will contact you shortly at {state['handoff_email']} or {state['handoff_phone']}. Can I help you with anything else?"
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response handoff_completed ticket_id=%s email=%s phone=%s", 
                          ticket_id, state["handoff_email"], state["handoff_phone"])
                return state
            else:
                reply = "I didn't catch a valid phone number. Please provide your phone number."
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response handoff_phone_invalid=1")
                return state
        elif handoff_state == "awaiting_confirmation":
            # User is confirming they want to talk to human
            user_text_lower = user_text.lower()
            if any(word in user_text_lower for word in ["yes", "yeah", "yep", "sure", "ok", "okay", "confirm", "proceed", "connect me"]) or "confirm_handoff" in user_text:
                state["handoff_state"] = "collecting_email"
                reply = "Great! To connect you with a human agent, I'll need your email address first."
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response handoff_confirmed=1")
                return state
            elif any(word in user_text_lower for word in ["no", "nope", "cancel", "cancel_handoff"]) or "cancel_handoff" in user_text:
                # User declined or changed mind
                state["handoff_state"] = None
                reply = "No problem! How else can I help you today?"
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response handoff_declined=1")
                return state
        else:
            # First time - ask for confirmation
            state["handoff_state"] = "awaiting_confirmation"
            reply = "I understand you'd like to speak with a human agent. Would you like me to connect you with one?"
            buttons = [
                {"label": "Yes", "action": "confirm_handoff", "type": "primary"},
                {"label": "No", "action": "cancel_handoff", "type": "secondary"},
            ]
            state["button_suggestions"] = buttons
            state["messages"].append({"role": "assistant", "content": reply})
            logger.info("node=response handoff_initiated=1")
            return state

    if missing == "auth":
        reply = (
            "I can look up your order. Please share the email you used and the order number (like BLB-1001). "
            "After that, I'll tell you the exact status. Can I help you with anything else?"
        )
        state["messages"].append({"role": "assistant", "content": reply})
        logger.info("node=response auth_prompted=1")
        return state

    # Check if user just provided auth details and we should fetch order status
    # This handles the case where user provides email/order but intent wasn't detected as order_status
    if customer_authed and customer_email and order_number:
        # Check if user is asking about order status (current intent or in conversation history)
        # Look at recent user messages to see if they asked about order status
        recent_user_messages = [m.get("content", "").lower() for m in state["messages"][-5:] if m.get("role") == "user"]
        is_order_related = (
            intent == "order_status" or 
            "order" in user_text.lower() or 
            ("where" in user_text.lower() and "order" in user_text.lower()) or
            any("order" in msg or ("where" in msg and "order" in msg) for msg in recent_user_messages)
        )
        
        # If we have auth but no tool_result yet (or error), and user is asking about orders, fetch it now
        if is_order_related and (not tool_result or (isinstance(tool_result, dict) and tool_result.get("error"))):
            logger.info("node=response fetching_order_status_post_auth email=%s order=%s", customer_email, order_number)
            from tools import get_order_status
            tool_result = get_order_status(customer_email, order_number)
            state["tool_result"] = tool_result
            # Update intent to order_status if it wasn't already
            if intent != "order_status":
                state["intent"] = "order_status"

    # For order_status, compose a direct concise message including items and shipping
    if intent == "order_status" and isinstance(tool_result, dict) and not tool_result.get("error"):
        items = tool_result.get("items", [])
        items_str = ", ".join([f"{it.get('name')} (Qty: {it.get('quantity', 1)})" for it in items]) if items else ""
        parts = []
        parts.append(f"Order {tool_result.get('order_number')} is {tool_result.get('status')}")
        if items_str:
            parts.append(f"Items: {items_str}")
        if tool_result.get("carrier") and tool_result.get("tracking_id"):
            parts.append(f"Shipped via {tool_result.get('carrier')} (Tracking: {tool_result.get('tracking_id')})")
        if tool_result.get("eta"):
            parts.append(f"ETA: {tool_result.get('eta')}")
        if tool_result.get("shipping_address"):
            parts.append(f"Ship-to: {tool_result.get('shipping_address')}")
        reply = ". ".join(parts) + ".\n\nCan I help you with anything else?"
        state["messages"].append({"role": "assistant", "content": reply})
        logger.info("node=response composed_order_status reply_len=%d", len(reply))
        return state
    
    # Handle order_status errors
    if intent == "order_status" and isinstance(tool_result, dict) and tool_result.get("error"):
        error = tool_result.get("error")
        if error == "customer_not_found":
            reply = "I couldn't find an account with that email. Please check and try again. Can I help you with anything else?"
        elif error == "order_not_found":
            reply = f"I couldn't find order {order_number} for that email. Please verify your order number and try again. Can I help you with anything else?"
        else:
            reply = "I'm having trouble looking up your order right now. Please try again later. Can I help you with anything else?"
        state["messages"].append({"role": "assistant", "content": reply})
        logger.info("node=response order_status_error=%s", error)
        return state

    # Handle questions about recommended products
    if intent == "product_question":
        last_recs = state.get("last_recommendations", [])
        if last_recs:
            # Extract information directly from product data to avoid LLM hallucination
            user_text_lower = user_text.lower()
            
            # Check if user is asking for NEW recommendations based on specific notes or product type
            # Patterns like "with lemon", "anything with rose", "single product not package"
            note_request_patterns = [
                "with ", "anything with", "something with", "perfume with",
                "that has", "which has", "containing"
            ]
            is_note_request = any(pattern in user_text_lower for pattern in note_request_patterns)
            
            # Check if user is changing product type (single product vs package/gift set)
            single_product_patterns = [
                "single product", "single perfume", "single bottle", "individual product",
                "not package", "no package", "not a package", "not collection", "no collection"
            ]
            package_patterns = [
                "package", "collection", "gift set", "set", "bundle"
            ]
            is_single_product_request = any(pattern in user_text_lower for pattern in single_product_patterns)
            is_package_request = any(pattern in user_text_lower for pattern in package_patterns) and not is_single_product_request
            
            # If user is asking for products with specific notes or changing product type, treat as new product_recommendation
            if is_note_request or is_single_product_request or is_package_request:
                # Re-route to product_recommendation flow to find new matches
                state["intent"] = "product_recommendation"
                logger.info("node=response re_routing_note_request_to_recommendation")
                # Let the recommendation flow handle this
                # Clear awaiting_choice to start fresh
                state["awaiting_choice"] = None
                # Continue to product_recommendation flow below
                intent = "product_recommendation"
            else:
                # Try to find product by name if mentioned - check both in last_recs and catalog
                product = None
                
                # First, check if user is asking about a product in last_recs
                for rec in last_recs:
                    rec_name = rec.get("name", "").lower()
                    # Remove size info for better matching
                    rec_name_clean = rec_name.replace(" - 100ml", "").replace(" - 50ml", "").strip()
                    # Get significant words (length > 2, not common words)
                    rec_words = [w for w in rec_name_clean.split() if len(w) > 2 and w not in ["the", "and", "for"]]
                    
                    # Check if significant words from product name appear in query
                    if rec_words and any(word in user_text_lower for word in rec_words):
                        product = rec
                        logger.info("node=response matched_product_from_last_recs product_name=%s", rec.get("name"))
                        break
                
                # If no match in last_recs, try searching the entire catalog
                # This handles cases where user asks about a different product
                if not product:
                    catalog_product = get_product_by_name(user_text)
                    if catalog_product:
                        # Check if this is a different product than what's in last_recs
                        catalog_id = catalog_product.get("id")
                        is_different_product = not any(rec.get("id") == catalog_id for rec in last_recs)
                        
                        if is_different_product:
                            logger.info("node=response found_different_product_in_catalog product_name=%s", catalog_product.get("name"))
                            # User is asking about a different product - update last_recommendations
                            state["last_recommendations"] = [catalog_product]
                            last_recs = [catalog_product]
                            product = catalog_product
                        else:
                            # Same product, use it
                            product = catalog_product
                            logger.info("node=response found_same_product_in_catalog product_name=%s", catalog_product.get("name"))
                
                # If still no specific product found, use first one from last_recs
                if not product:
                    product = last_recs[0] if last_recs else {}
                    logger.info("node=response using_first_from_last_recs product_name=%s", product.get("name"))
                
                # Handle price questions
                if any(word in user_text_lower for word in ["price", "cost", "how much"]):
                    price = product.get("price", 0)
                    name = product.get("name", "this product")
                    if price:
                        reply = f"The price of {name} is ₹{price}. Want to see it or explore other options?"
                    else:
                        reply = f"I don't have the price information for {name} right now. Can I help you with anything else?"
                    
                    # Add action buttons
                    buttons = []
                    product_url = product.get("product_url", "")
                    if product_url:
                        buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
                    
                    # Add other options button if more recommendations available
                    current_index = state.get("recommendation_index", 0)
                    remaining = len(last_recs) - current_index - 1
                    if remaining > 0:
                        buttons.append({"label": "Other Options", "action": "show_other", "type": "primary"})
                    
                    # Add safer/bolder buttons
                    current_price = product.get("price", 0)
                    safer_exists = any(p.get("price", 0) < current_price for p in last_recs if p.get("id") != product.get("id"))
                    if safer_exists:
                        buttons.append({"label": "Show Safer Option", "action": "show_safer", "type": "secondary"})
                    
                    bolder_exists = any(p.get("price", 0) > current_price for p in last_recs if p.get("id") != product.get("id"))
                    if bolder_exists:
                        buttons.append({"label": "Show Bolder Option", "action": "show_bolder", "type": "secondary"})
                    
                    if buttons:
                        state["button_suggestions"] = buttons
                    
                    state["messages"].append({"role": "assistant", "content": reply})
                    logger.info("node=response answered_price_question=1 buttons=%d", len(buttons))
                    return state
                
                # Handle occasion/when to wear questions
                if any(phrase in user_text_lower for phrase in ["when", "wear", "occasion", "use", "suitable"]):
                    name = product.get("name", "this product")
                    occasion_tags = product.get("occasion_tags", [])
                    vibe_tags = product.get("vibe_tags", [])
                    
                    reply_parts = [f"{name} is great for"]
                    
                    if occasion_tags:
                        # Format occasions nicely
                        occasions = [occ.replace("_", " ") for occ in occasion_tags]
                        if len(occasions) == 1:
                            reply_parts.append(occasions[0])
                        elif len(occasions) == 2:
                            reply_parts.append(f"{occasions[0]} and {occasions[1]}")
                        else:
                            reply_parts.append(f"{', '.join(occasions[:-1])}, and {occasions[-1]}")
                    else:
                        reply_parts.append("everyday wear")
                    
                    # Add vibe context
                    if vibe_tags:
                        vibe_desc = ", ".join(vibe_tags[:3])
                        reply_parts.append(f"It has {vibe_desc} vibes.")
                    
                    reply = " ".join(reply_parts) + " Want to see it or explore other options?"
                    
                    # Add action buttons
                    buttons = []
                    product_url = product.get("product_url", "")
                    if product_url:
                        buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
                    
                    # Add other options button if more recommendations available
                    current_index = state.get("recommendation_index", 0)
                    remaining = len(last_recs) - current_index - 1
                    if remaining > 0:
                        buttons.append({"label": "Other Options", "action": "show_other", "type": "primary"})
                    
                    # Add safer/bolder buttons
                    current_price = product.get("price", 0)
                    current_vibes = product.get("vibe_tags", [])
                    
                    safer_candidates = [
                        p for p in last_recs 
                        if p.get("id") != product.get("id") and (
                            p.get("price", 0) < current_price or
                            any(v in p.get("vibe_tags", []) for v in ["fresh", "clean", "light", "breezy"])
                        )
                    ]
                    if safer_candidates:
                        buttons.append({"label": "Show Safer Option", "action": "show_safer", "type": "secondary"})
                    
                    bolder_candidates = [
                        p for p in last_recs 
                        if p.get("id") != product.get("id") and (
                            p.get("price", 0) > current_price or
                            any(v in p.get("vibe_tags", []) for v in ["bold", "intense", "strong", "oud", "spicy"])
                        )
                    ]
                    if bolder_candidates:
                        buttons.append({"label": "Show Bolder Option", "action": "show_bolder", "type": "secondary"})
                    
                    if buttons:
                        state["button_suggestions"] = buttons
                    
                    state["messages"].append({"role": "assistant", "content": reply})
                    logger.info("node=response answered_occasion_question=1 buttons=%d", len(buttons))
                    return state
                
                # Handle questions about contents/what's included
                if any(phrase in user_text_lower for phrase in ["what do i get", "what's in", "what is in", "what comes", "contents", "includes"]):
                    # Use the product found above, or first one
                    if not product:
                        product = last_recs[0] if last_recs else {}
                    name = product.get("name", "this product")
                    desc = product.get("short_description", "")
                    notes = product.get("notes", {})
                    
                    reply_parts = [f"{name}"]
                    if desc:
                        reply_parts.append(f"includes: {desc}")
                    
                    # Extract notes if available
                    all_notes = []
                    if notes:
                        all_notes = (
                            notes.get("head", []) + 
                            notes.get("heart", []) + 
                            notes.get("base", [])
                        )
                    
                    if all_notes:
                        notes_str = ", ".join(all_notes[:5])  # Limit to first 5 notes
                        reply_parts.append(f"Key notes: {notes_str}")
                    
                    reply = ". ".join(reply_parts) + ". Want to see it or explore other options?"
                    
                    # Add action buttons
                    buttons = []
                    product_url = product.get("product_url", "")
                    if product_url:
                        buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
                    
                    current_index = state.get("recommendation_index", 0)
                    remaining = len(last_recs) - current_index - 1
                    if remaining > 0:
                        buttons.append({"label": "Other Options", "action": "show_other", "type": "primary"})
                    
                    if buttons:
                        state["button_suggestions"] = buttons
                    
                    state["messages"].append({"role": "assistant", "content": reply})
                    logger.info("node=response answered_contents_question=1 buttons=%d", len(buttons))
                    return state
                
                # Handle other questions - use LLM but with structured product data
                import json
                products_context = json.dumps(last_recs[:2], indent=2)
                
                # Get conversation context (recent user messages for occasion/context)
                user_messages = [m["content"] for m in state["messages"] if m["role"] == "user"]
                conversation_context = " ".join(user_messages[-3:]) if len(user_messages) > 1 else ""
                
                # Get user preferences for context
                prefs = state.get("preferences") or {}
                occasion_context = ""
                if prefs.get("occasion_raw"):
                    # Use raw occasion text (e.g., "wedding") for natural reference
                    occasion_context = f"\nUser mentioned occasion: {prefs.get('occasion_raw')}"
                elif prefs.get("occasions"):
                    occasions = prefs.get("occasions", [])
                    occasion_context = f"\nUser mentioned occasion: {', '.join(occasions)}"
                
                prompt = f"""{ASSISTANT_STYLE_PROMPT}

The user is asking about these recommended products:
{products_context}

Conversation context: {conversation_context}
{occasion_context}

User question: {user_text}

IMPORTANT: 
- Extract information directly from the product data above. Do not make up prices or details.
- Price is in the "price" field (in rupees ₹)
- Description is in "short_description"
- Notes are in "notes" object with "head", "heart", "base" arrays
- Name is in "name" field
- Reference the conversation context (like wedding, occasion, etc.) in your answer to make it relevant
- DO NOT use markdown formatting, asterisks (**), or bold text - write in plain text only
- Keep response SHORT (2-3 sentences max) and conversational
- If user mentioned specific occasion earlier (wedding, date, party), reference it naturally

Answer their question naturally using ONLY the data provided above.
End with: "Want to see it or explore other options?"
"""
                answer = _call_llm(prompt)
                
                # Add action buttons
                buttons = []
                current_index = state.get("recommendation_index", 0)
                if current_index < len(last_recs):
                    product = last_recs[current_index]
                    product_url = product.get("product_url", "")
                    if product_url:
                        buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
                
                remaining = len(last_recs) - current_index - 1
                if remaining > 0:
                    buttons.append({"label": "Other Options", "action": "show_other", "type": "primary"})
                
                if buttons:
                    state["button_suggestions"] = buttons
                
                state["messages"].append({"role": "assistant", "content": answer})
                logger.info("node=response answered_product_question=1 buttons=%d", len(buttons))
                return state
        else:
            # No last recommendations - try to find product in catalog by name
            product = get_product_by_name(user_text)
            if product:
                logger.info("node=response found_product_in_catalog product_name=%s", product.get("name"))
                # Add to last_recommendations so it can be referenced
                state["last_recommendations"] = [product]
                last_recs = [product]
                
                # Extract information directly from product data
                user_text_lower = user_text.lower()
                
                # Handle price questions
                if any(word in user_text_lower for word in ["price", "cost", "how much"]):
                    price = product.get("price", 0)
                    name = product.get("name", "this product")
                    if price:
                        reply = f"The price of {name} is ₹{price}. Want to see it or explore similar options?"
                    else:
                        reply = f"I don't have the price information for {name} right now. Can I help you with anything else?"
                    
                    # Add action buttons
                    buttons = []
                    product_url = product.get("product_url", "")
                    if product_url:
                        buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
                    
                    buttons.append({"label": "Find My Perfume", "action": "find_perfume", "type": "primary"})
                    
                    if buttons:
                        state["button_suggestions"] = buttons
                    
                    state["messages"].append({"role": "assistant", "content": reply})
                    logger.info("node=response answered_price_question_from_catalog=1 buttons=%d", len(buttons))
                    return state
                
                # Handle occasion/when to wear questions
                if any(phrase in user_text_lower for phrase in ["when", "wear", "occasion", "use", "suitable"]):
                    name = product.get("name", "this product")
                    occasion_tags = product.get("occasion_tags", [])
                    vibe_tags = product.get("vibe_tags", [])
                    
                    reply_parts = [f"{name} is great for"]
                    
                    if occasion_tags:
                        # Format occasions nicely
                        occasions = [occ.replace("_", " ") for occ in occasion_tags]
                        if len(occasions) == 1:
                            reply_parts.append(occasions[0])
                        elif len(occasions) == 2:
                            reply_parts.append(f"{occasions[0]} and {occasions[1]}")
                        else:
                            reply_parts.append(f"{', '.join(occasions[:-1])}, and {occasions[-1]}")
                    else:
                        reply_parts.append("everyday wear")
                    
                    # Add vibe context
                    if vibe_tags:
                        vibe_desc = ", ".join(vibe_tags[:3])
                        reply_parts.append(f"It has {vibe_desc} vibes.")
                    
                    reply = " ".join(reply_parts) + " Want to see it or explore similar options?"
                    
                    # Add action buttons
                    buttons = []
                    product_url = product.get("product_url", "")
                    if product_url:
                        buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
                    
                    buttons.append({"label": "Find My Perfume", "action": "find_perfume", "type": "primary"})
                    
                    if buttons:
                        state["button_suggestions"] = buttons
                    
                    state["messages"].append({"role": "assistant", "content": reply})
                    logger.info("node=response answered_occasion_question_from_catalog=1 buttons=%d", len(buttons))
                    return state
                
                # Handle questions about contents/what's included/notes
                if any(phrase in user_text_lower for phrase in ["what do i get", "what's in", "what is in", "what comes", "contents", "includes", "notes", "what notes"]):
                    name = product.get("name", "this product")
                    desc = product.get("short_description", "")
                    notes = product.get("notes", {})
                    
                    reply_parts = [f"{name}"]
                    if desc:
                        reply_parts.append(f"includes: {desc}")
                    
                    # Extract notes if available
                    all_notes = []
                    if notes:
                        all_notes = (
                            notes.get("head", []) + 
                            notes.get("heart", []) + 
                            notes.get("base", [])
                        )
                    
                    if all_notes:
                        notes_str = ", ".join(all_notes[:5])  # Limit to first 5 notes
                        reply_parts.append(f"Key notes: {notes_str}")
                    
                    reply = ". ".join(reply_parts) + ". Want to see it or explore similar options?"
                    
                    # Add action buttons
                    buttons = []
                    product_url = product.get("product_url", "")
                    if product_url:
                        buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
                    
                    buttons.append({"label": "Find My Perfume", "action": "find_perfume", "type": "primary"})
                    
                    if buttons:
                        state["button_suggestions"] = buttons
                    
                    state["messages"].append({"role": "assistant", "content": reply})
                    logger.info("node=response answered_contents_question_from_catalog=1 buttons=%d", len(buttons))
                    return state
                
                # For other questions, use LLM with product data
                import json
                product_context = json.dumps(product, indent=2)
                
                prompt = f"""{ASSISTANT_STYLE_PROMPT}

The user is asking about this product:
{product_context}

User question: {user_text}

IMPORTANT: 
- Extract information directly from the product data above. Do not make up prices or details.
- Price is in the "price" field (in rupees ₹)
- Description is in "short_description"
- Notes are in "notes" object with "head", "heart", "base" arrays
- Name is in "name" field
- DO NOT use markdown formatting, asterisks (**), or bold text - write in plain text only
- Keep response SHORT (2-3 sentences max) and conversational

Answer their question naturally using ONLY the data provided above.
End with: "Want to see it or explore similar options?"
"""
                answer = _call_llm(prompt)
                
                # Add action buttons
                buttons = []
                product_url = product.get("product_url", "")
                if product_url:
                    buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
                
                buttons.append({"label": "Find My Perfume", "action": "find_perfume", "type": "primary"})
                
                if buttons:
                    state["button_suggestions"] = buttons
                
                state["messages"].append({"role": "assistant", "content": answer})
                logger.info("node=response answered_product_question_from_catalog=1 buttons=%d", len(buttons))
                return state
            else:
                # Product not found in catalog either
                reply = "I don't have any products in context right now. Could you tell me what you're looking for? Can I help you with anything else?"
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response product_question_no_context=1")
                return state

    # Handle BYOB flow
    if intent == "build_own_box":
        if isinstance(tool_result, dict) and tool_result.get("byob_products"):
            byob_products = tool_result["byob_products"]
            
            # Check if user has already started selecting
            selections = state.get("byob_selections") or []
            max_items = state.get("byob_max_items", 3)
            
            # Check if we're awaiting a BYOB selection and user just selected a product
            # Use the saved value from before clearing
            awaiting_byob = awaiting_choice_before_clear == "byob_selection"
            logger.info("node=response byob_check awaiting=%s user_text=%s selections=%s", awaiting_byob, user_text, selections)
            
            if awaiting_byob:
                # Try to match user's message to a BYOB product name
                user_text_lower = user_text.lower().strip()
                selected_product_id = None
                
                # First check if message contains "select_byob:" action format
                if "select_byob:" in user_text:
                    try:
                        selected_product_id = user_text.split("select_byob:")[1].strip()
                        logger.info("node=response byob_found_action_format product_id=%s", selected_product_id)
                    except Exception as e:
                        logger.error("node=response byob_action_parse_error error=%s", e)
                        pass
                
                # If not found, try matching by product name
                if not selected_product_id:
                    logger.info("node=response byob_matching_by_name user_text_lower=%s products_count=%d", user_text_lower, len(byob_products))
                    for product in byob_products:
                        product_name = product.get("name", "").lower().strip()
                        product_id = product.get("id", "")
                        # More flexible matching: check if user text matches product name
                        # Remove common punctuation and extra spaces for better matching
                        product_name_clean = product_name.replace("-", " ").replace("  ", " ")
                        user_text_clean = user_text_lower.replace("-", " ").replace("  ", " ")
                        
                        # Check multiple matching strategies
                        if (product_name == user_text_lower or 
                            user_text_lower == product_name or
                            user_text_clean == product_name_clean or
                            user_text_lower in product_name or 
                            product_name in user_text_lower or
                            user_text_clean in product_name_clean or
                            product_name_clean in user_text_clean):
                            selected_product_id = product_id
                            logger.info("node=response byob_matched_product product_id=%s product_name=%s", product_id, product_name)
                            break
                
                if not selected_product_id:
                    logger.warning("node=response byob_no_match user_text=%s", user_text)
                
                # If we found a selection and it's not already selected
                if selected_product_id and selected_product_id not in selections:
                    selections.append(selected_product_id)
                    state["byob_selections"] = selections
                    logger.info("node=response byob_product_selected=1 product_id=%s total_selected=%d", selected_product_id, len(selections))
                    
                    # Check if we've reached the max
                    if len(selections) >= max_items:
                        # User has completed selection
                        validation = validate_byob_selection(selections, max_items)
                        if validation.get("valid"):
                            selected_products = validation["products"]
                            total_price = validation["total_price"]
                            
                            # Build summary
                            product_names = [p.get("name", "") for p in selected_products]
                            reply = f"Perfect! Your custom box includes: {', '.join(product_names)}. "
                            reply += f"Total: ₹{total_price}. "
                            reply += "Visit our BYOB page to complete your order. Can I help you with anything else?"
                            
                            # Add BYOB page link button
                            buttons = [{"label": "View BYOB Page", "action": "open_url:https://blabliblulife.com/pages/build-your-own-box", "type": "success"}]
                            state["button_suggestions"] = buttons
                        else:
                            reply = "There was an issue with your selections. Let's start over. Can I help you with anything else?"
                        
                        state["messages"].append({"role": "assistant", "content": reply})
                        # Clear selections after completion
                        state["byob_selections"] = []
                        state["awaiting_choice"] = None
                        logger.info("node=response byob_completed=1")
                        return state
                    else:
                        # Still need more selections - show updated list
                        remaining = max_items - len(selections)
                        reply = f"Great! You've selected {len(selections)}. Choose {remaining} more. "
                elif selected_product_id in selections:
                    # Product already selected
                    remaining = max_items - len(selections)
                    reply = f"You've already selected that one. You've selected {len(selections)}. Choose {remaining} more. "
                else:
                    # Couldn't match the selection - show products again
                    remaining = max_items - len(selections)
                    if len(selections) == 0:
                        reply = f"Great! You can choose {max_items} perfumes for your custom box. "
                    else:
                        reply = f"You've selected {len(selections)}. Choose {remaining} more. "
            else:
                # First time showing products
                remaining = max_items - len(selections)
                if len(selections) == 0:
                    reply = f"Great! You can choose {max_items} perfumes for your custom box. "
                else:
                    reply = f"You've selected {len(selections)}. Choose {remaining} more. "
            
            # Generate button suggestions for selection (exclude already selected)
            buttons = []
            for product in byob_products[:8]:  # Show top 8 options
                pid = product.get("id", "")
                name = product.get("name", "")
                if pid not in selections:
                    buttons.append({
                        "label": name,
                        "action": f"select_byob:{pid}",
                        "type": "primary"
                    })
            
            state["button_suggestions"] = buttons
            state["awaiting_choice"] = "byob_selection"
            
            reply += "Select your perfumes from the options below."
            state["messages"].append({"role": "assistant", "content": reply})
            logger.info("node=response byob_awaiting_selection remaining=%d selected=%d", remaining, len(selections))
            return state
        else:
            reply = "I couldn't load the BYOB products right now. Please try again. Can I help you with anything else?"
            state["messages"].append({"role": "assistant", "content": reply})
            logger.info("node=response byob_error=no_products")
            return state
    
    # Handle general questions (FAQ)
    if intent == "general_question":
        if isinstance(tool_result, dict) and tool_result.get("faq_answer"):
            reply = tool_result["faq_answer"] + " Can I help you with anything else?"
            state["messages"].append({"role": "assistant", "content": reply})
            logger.info("node=response general_question_answered=1")
            return state
        else:
            # No FAQ match - offer human handoff
            reply = "I'm not sure about that. Would you like to talk with a human agent who can help you better?"
            buttons = [
                {"label": "Yes, connect me", "action": "confirm_handoff", "type": "primary"},
                {"label": "No, thanks", "action": "cancel_handoff", "type": "secondary"},
            ]
            state["button_suggestions"] = buttons
            state["handoff_state"] = "awaiting_confirmation"
            state["messages"].append({"role": "assistant", "content": reply})
            logger.info("node=response general_question_no_match offering_handoff=1")
            return state
    
    # For recommendations: ask preferences one by one, remember from context
    if intent in ("product_recommendation", "find_perfume", "gift_recommendation"):
        # Check if we have sufficient preferences before recommending
        prefs = state.get("preferences") or {}
        has_gender = prefs.get("gender") is not None
        has_scent_type = isinstance(prefs.get("scent_types"), list) and len(prefs.get("scent_types", [])) > 0
        has_budget = prefs.get("budget") is not None
        
        # If intent is gift_recommendation, mark as gift
        is_gift = prefs.get("is_gift", False) or intent == "gift_recommendation"
        if intent == "gift_recommendation" and not prefs.get("is_gift"):
            prefs["is_gift"] = True
            state["preferences"] = prefs
        
        # Initialize questions_asked if not present
        if "questions_asked" not in state:
            state["questions_asked"] = []
        
        questions_asked = state.get("questions_asked", [])
        
        # If gender was auto-detected (e.g., from "girlfriend", "boyfriend"), mark it as asked
        if has_gender and "gender" not in questions_asked:
            questions_asked.append("gender")
            state["questions_asked"] = questions_asked
        
        has_occasion = isinstance(prefs.get("occasions"), list) and len(prefs.get("occasions", [])) > 0
        
        # For gift queries, follow the flow from bot_suggestion_flow.txt:
        # Ask gender -> vibe (personality) -> budget (3 questions before recommending)
        # For regular queries, follow the flow from bot_suggestion_flow.txt:
        # Ask gender -> vibe -> occasion -> budget (all 4 questions before recommending)
        if is_gift:
            # Gifts need: gender, vibe (personality), budget
            # Check if these specific questions have been asked
            gift_required_questions = {"gender", "vibe", "budget"}
            has_sufficient_prefs = gift_required_questions.issubset(set(questions_asked))
        else:
            # Only recommend after ALL 4 questions have been asked
            # Questions: gender, vibe, occasion, budget
            required_questions = {"gender", "vibe", "occasion", "budget"}
            has_sufficient_prefs = required_questions.issubset(set(questions_asked))
        
        # Only recommend if we have sufficient preferences AND recommendations exist
        if has_sufficient_prefs:
            if isinstance(tool_result, dict) and tool_result.get("recommendations"):
                recs = tool_result["recommendations"]
                
                # Check what the user is asking for
                user_text_lower = user_text.lower()
                asking_for_safer = any(phrase in user_text_lower for phrase in ["safer option", "safer", "lighter", "softer"])
                asking_for_bolder = any(phrase in user_text_lower for phrase in ["bolder option", "bolder", "stronger", "more intense"])
                asking_for_other = any(phrase in user_text_lower for phrase in [
                    "other option", "another option", "other options", "anything else", "something else",
                    "show me another", "different one", "what else", "more options"
                ])
                
                # Get current recommendation index (default to 0 if not set)
                current_index = state.get("recommendation_index", 0)
                current_product = recs[current_index] if current_index < len(recs) else None
                
                # Handle safer/bolder/other options
                # Get user's gender preference to filter recommendations
                user_gender = prefs.get("gender")
                
                if asking_for_safer and current_product:
                    # Find the safest option (lowest price or lightest vibe) matching user's gender
                    current_price = current_product.get("price", 0)
                    safer_recs = [
                        (idx, p) for idx, p in enumerate(recs)
                        if idx != current_index and (
                            # Match gender if specified, or accept unisex
                            (not user_gender or p.get("gender_profile") == user_gender or p.get("gender_profile") == "unisex")
                        ) and (
                            p.get("price", 0) < current_price or
                            any(v in p.get("vibe_tags", []) for v in ["fresh", "clean", "light", "breezy"])
                        )
                    ]
                    if safer_recs:
                        # Sort by price (ascending) and pick the first
                        safer_recs.sort(key=lambda x: x[1].get("price", 0))
                        current_index = safer_recs[0][0]
                elif asking_for_bolder and current_product:
                    # Find the boldest option (highest price or strongest vibe) matching user's gender
                    current_price = current_product.get("price", 0)
                    bolder_recs = [
                        (idx, p) for idx, p in enumerate(recs)
                        if idx != current_index and (
                            # Match gender if specified, or accept unisex
                            (not user_gender or p.get("gender_profile") == user_gender or p.get("gender_profile") == "unisex")
                        ) and (
                            p.get("price", 0) > current_price or
                            any(v in p.get("vibe_tags", []) for v in ["bold", "intense", "strong", "oud", "spicy"])
                        )
                    ]
                    if bolder_recs:
                        # Sort by price (descending) and pick the first
                        bolder_recs.sort(key=lambda x: x[1].get("price", 0), reverse=True)
                        current_index = bolder_recs[0][0]
                elif asking_for_other:
                    # Just show the next product in the list
                    current_index += 1
                
                # Check if we have more recommendations to show
                if current_index < len(recs):
                    top = recs[current_index]
                    state["last_recommendations"] = recs  # Keep all recommendations
                    state["recommendation_index"] = current_index  # Update index
                    
                    name = top.get("name", "")
                    positioning = top.get("positioning_line", "")
                    desc = top.get("short_description", "")
                    price = top.get("price", 0)
                    
                    # Build a natural response with context awareness
                    reply_parts = []
                    
                    # Add context prefix based on what user selected
                    if asking_for_other:
                        reply_parts.append(f"Here's another option: {name}.")
                    elif asking_for_bolder:
                        reply_parts.append(f"Here's a bolder option: {name}.")
                    elif asking_for_safer:
                        reply_parts.append(f"Here's a safer option: {name}.")
                    else:
                        reply_parts.append(f"I'd suggest {name}.")
                    
                    # Keep response short - combine positioning and description
                    if positioning:
                        reply_parts.append(positioning)
                    elif desc:
                        reply_parts.append(desc)
                    
                    if price:
                        reply_parts.append(f"Priced at ₹{price}.")
                    
                    # Add button suggestions for safer/bolder options and other actions
                    buttons = []
                    remaining = len(recs) - current_index - 1
                    
                    # Add safer/bolder option buttons based on product characteristics
                    # Safer = lower price or lighter scent, Bolder = higher price or stronger scent
                    current_price = top.get("price", 0)
                    current_vibes = top.get("vibe_tags", [])
                    
                    # Find safer option (lighter/cheaper) matching user's gender
                    safer_candidates = [
                        (idx, p) for idx, p in enumerate(recs) 
                        if idx != current_index and (
                            # Match gender if specified, or accept unisex
                            (not user_gender or p.get("gender_profile") == user_gender or p.get("gender_profile") == "unisex")
                        ) and (
                            p.get("price", 0) < current_price or
                            any(v in p.get("vibe_tags", []) for v in ["fresh", "clean", "light", "breezy"])
                        )
                    ]
                    if safer_candidates:
                        buttons.append({"label": "Show Safer Option", "action": "show_safer", "type": "secondary"})
                    
                    # Find bolder option (stronger/pricier) matching user's gender
                    bolder_candidates = [
                        (idx, p) for idx, p in enumerate(recs) 
                        if idx != current_index and (
                            # Match gender if specified, or accept unisex
                            (not user_gender or p.get("gender_profile") == user_gender or p.get("gender_profile") == "unisex")
                        ) and (
                            p.get("price", 0) > current_price or
                            any(v in p.get("vibe_tags", []) for v in ["bold", "intense", "strong", "oud", "spicy"])
                        )
                    ]
                    if bolder_candidates:
                        buttons.append({"label": "Show Bolder Option", "action": "show_bolder", "type": "secondary"})
                    
                    # Add "other options" button if more products available
                    if remaining > 0:
                        buttons.append({"label": "Other Options", "action": "show_other", "type": "primary"})
                        reply_parts.append("Want to know more or see other options?")
                    else:
                        reply_parts.append("Want to know more?")
                    
                    # Add "View Product" button
                    product_url = top.get("product_url", "")
                    if product_url:
                        buttons.append({"label": "View Product", "action": f"open_url:{product_url}", "type": "success"})
                    
                    if buttons:
                        state["button_suggestions"] = buttons
                    
                    reply = " ".join(reply_parts)
                    
                    state["messages"].append({"role": "assistant", "content": reply})
                    logger.info("node=response used_recommendations=1 showing_product_index=%d remaining=%d buttons=%d", 
                               current_index, remaining, len(buttons))
                    return state
                else:
                    # No more recommendations
                    total_shown = len(recs)
                    if total_shown == 1:
                        reply = "That's the only option matching your preferences. Would you like to adjust your requirements?"
                    else:
                        reply = f"Those are the {total_shown} options matching your preferences. Would you like to adjust your requirements to see more?"
                    state["messages"].append({"role": "assistant", "content": reply})
                    logger.info("node=response exhausted_recommendations total_shown=%d", total_shown)
                    return state
            else:
                # Sufficient preferences but NO recommendations found (e.g. strict budget filter)
                # Check what might be the blocker
                buttons = []
                reply = "I couldn't find any perfumes matching your exact criteria."
                
                if has_budget:
                    # Offer budget adjustment options
                    reply += " It might be the budget constraint. Would you like to see options with a slightly higher budget?"
                    
                    # Set awaiting_choice so collect_preferences knows user is adjusting budget
                    state["awaiting_choice"] = "budget"
                    
                    # Get current budget
                    budget_info = prefs.get("budget", {})
                    if isinstance(budget_info, dict):
                        current_amount = budget_info.get("amount", 0)
                        operator = budget_info.get("operator", "under")
                        
                        # Offer higher budget options
                        if operator == "under":
                            if current_amount < 700:
                                buttons.append({"label": "Under ₹700", "action": "select_budget:under_700", "type": "primary"})
                            if current_amount < 900:
                                buttons.append({"label": "₹700-₹900", "action": "select_budget:700_900", "type": "primary"})
                            buttons.append({"label": "No Budget Limit", "action": "select_budget:no_limit", "type": "primary"})
                        
                        # Also offer to change other preferences
                        buttons.append({"label": "Change Vibe", "action": "change_vibe", "type": "secondary"})
                    else:
                        # Fallback buttons
                        buttons = [
                            {"label": "Under ₹700", "action": "select_budget:under_700", "type": "primary"},
                            {"label": "₹700-₹900", "action": "select_budget:700_900", "type": "primary"},
                            {"label": "No Budget Limit", "action": "select_budget:no_limit", "type": "primary"},
                        ]
                else:
                    # Offer to adjust other preferences
                    reply += " Would you like to adjust your preferences?"
                    buttons = [
                        {"label": "Change Gender", "action": "change_gender", "type": "primary"},
                        {"label": "Change Vibe", "action": "change_vibe", "type": "primary"},
                    ]
                    if not is_gift:
                        buttons.append({"label": "Change Occasion", "action": "change_occasion", "type": "primary"})
                
                if buttons:
                    state["button_suggestions"] = buttons
                
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response no_recommendations_found sufficient_prefs=1 buttons=%d", len(buttons))
                return state
        
        # Ask for missing preferences ONE AT A TIME with button suggestions
        # Priority: gender > vibe/scent_type > occasion > budget
        buttons = []
        
        # Use saved awaiting_choice for preference detection
        current_awaiting = awaiting_choice_before_clear
        
        if is_gift:
            # Gift flow: ask gender -> vibe (personality) -> budget
            if not has_gender:
                reply = "Who is this gift for?"
                buttons = [
                    {"label": "For Him", "action": "select_gender:for_him", "type": "primary"},
                    {"label": "For Her", "action": "select_gender:for_her", "type": "primary"},
                    {"label": "Unisex", "action": "select_gender:unisex", "type": "secondary"},
                ]
                state["awaiting_choice"] = "gender"
                if "gender" not in questions_asked:
                    questions_asked.append("gender")
                    state["questions_asked"] = questions_asked
            elif "vibe" not in questions_asked:
                # Have gender, ask about recipient's personality (maps to vibe/scent)
                # Use appropriate pronouns based on gift recipient gender
                gender = prefs.get("gender")
                if gender == "for_him":
                    reply = "What's his personality like?"
                elif gender == "for_her":
                    reply = "What's her personality like?"
                else:
                    reply = "What's their personality like?"
                
                buttons = [
                    {"label": "Classy & Elegant", "action": "select_vibe:floral", "type": "primary"},
                    {"label": "Playful & Fun", "action": "select_vibe:fresh", "type": "primary"},
                    {"label": "Sweet & Romantic", "action": "select_vibe:sweet", "type": "primary"},
                    {"label": "Intense & Bold", "action": "select_vibe:spicy", "type": "primary"},
                ]
                state["awaiting_choice"] = "vibe"
                questions_asked.append("vibe")
                state["questions_asked"] = questions_asked
            elif "budget" not in questions_asked:
                # Have gender and vibe, ask for budget
                reply = "What's your budget range?"
                buttons = [
                    {"label": "Under ₹700", "action": "select_budget:under_700", "type": "primary"},
                    {"label": "₹700-₹900", "action": "select_budget:700_900", "type": "primary"},
                    {"label": "No Limit", "action": "select_budget:no_limit", "type": "secondary"},
                ]
                state["awaiting_choice"] = "budget"
                questions_asked.append("budget")
                state["questions_asked"] = questions_asked
            else:
                # All gift questions asked, make recommendation
                reply = "Let me find the perfect gift set for you."
        else:
            # Regular product recommendation flow - ask ONE question at a time
            if not has_gender:
                reply = "Who is this for?"
                buttons = [
                    {"label": "For Him", "action": "select_gender:for_him", "type": "primary"},
                    {"label": "For Her", "action": "select_gender:for_her", "type": "primary"},
                    {"label": "Unisex", "action": "select_gender:unisex", "type": "secondary"},
                ]
                state["awaiting_choice"] = "gender"
                if "gender" not in questions_asked:
                    questions_asked.append("gender")
                    state["questions_asked"] = questions_asked
            elif not has_scent_type:
                # Have gender, ask for vibe/scent type next
                reply = "What kind of vibe are you looking for?"
                buttons = [
                    {"label": "Fresh & Breezy", "action": "select_vibe:fresh", "type": "primary"},
                    {"label": "Sweet & Gourmand", "action": "select_vibe:sweet", "type": "primary"},
                    {"label": "Spicy & Bold", "action": "select_vibe:spicy", "type": "primary"},
                    {"label": "Oud & Intense", "action": "select_vibe:oud", "type": "primary"},
                    {"label": "Soft & Cozy", "action": "select_vibe:floral", "type": "secondary"},
                    {"label": "Woody & Earthy", "action": "select_vibe:woody", "type": "secondary"},
                ]
                state["awaiting_choice"] = "vibe"
                if "vibe" not in questions_asked:
                    questions_asked.append("vibe")
                    state["questions_asked"] = questions_asked
            elif "occasion" not in questions_asked:
                # Have gender and scent, ask for occasion
                reply = "When will you wear it?"
                buttons = [
                    {"label": "Daily", "action": "select_occasion:casual", "type": "primary"},
                    {"label": "Office", "action": "select_occasion:formal", "type": "primary"},
                    {"label": "Date Night", "action": "select_occasion:date_night", "type": "primary"},
                    {"label": "Party", "action": "select_occasion:party", "type": "primary"},
                    {"label": "Mixed", "action": "select_occasion:mixed", "type": "secondary"},
                ]
                state["awaiting_choice"] = "occasion"
                questions_asked.append("occasion")
                state["questions_asked"] = questions_asked
            elif "budget" not in questions_asked:
                # Have all other prefs, ask for budget
                reply = "What's your budget range?"
                buttons = [
                    {"label": "Under ₹700", "action": "select_budget:under_700", "type": "primary"},
                    {"label": "₹700-₹900", "action": "select_budget:700_900", "type": "primary"},
                    {"label": "No Limit", "action": "select_budget:no_limit", "type": "secondary"},
                ]
                state["awaiting_choice"] = "budget"
                questions_asked.append("budget")
                state["questions_asked"] = questions_asked
            else:
                # All questions have been asked, make recommendation
                reply = "Let me find the perfect perfume for you."
        
        if buttons:
            state["button_suggestions"] = buttons
        
        state["messages"].append({"role": "assistant", "content": reply})
        logger.info("node=response ask_clarifying_for_reco=1 missing_gender=%s missing_scent=%s missing_budget=%s is_gift=%s buttons=%d", 
                   not has_gender, not has_scent_type, not has_budget, is_gift, len(buttons))
        return state

    # If product_info but no exact match, check if user is asking about previously recommended products
    if intent == "product_info":
        last_recs = state.get("last_recommendations", [])
        if last_recs:
            # User might be asking about a previously recommended product
            # Check if the question matches product_question patterns
            user_text_lower = user_text.lower()
            if any(phrase in user_text_lower for phrase in ["what", "price", "cost", "contents", "includes", "what's in", "what do i get"]):
                # Treat as product_question instead
                product = last_recs[0] if last_recs else {}
                price = product.get("price", 0)
                name = product.get("name", "this product")
                desc = product.get("short_description", "")
                notes = product.get("notes", {})
                
                if "price" in user_text_lower or "cost" in user_text_lower:
                    if price:
                        reply = f"The price of {name} is ₹{price}. Can I help you with anything else?"
                    else:
                        reply = f"I don't have the price information for {name} right now. Can I help you with anything else?"
                elif any(phrase in user_text_lower for phrase in ["what do i get", "what's in", "what is in", "contents", "includes"]):
                    reply_parts = [f"{name}"]
                    if desc:
                        reply_parts.append(f"includes: {desc}")
                    all_notes = []
                    if notes:
                        all_notes = (
                            notes.get("head", []) + 
                            notes.get("heart", []) + 
                            notes.get("base", [])
                        )
                    if all_notes:
                        notes_str = ", ".join(all_notes[:5])
                        reply_parts.append(f"Key notes: {notes_str}")
                    reply = ". ".join(reply_parts) + ". Can I help you with anything else?"
                else:
                    # Generic product info question
                    reply_parts = [f"{name}"]
                    if desc:
                        reply_parts.append(desc)
                    if price:
                        reply_parts.append(f"Priced at ₹{price}.")
                    reply = ". ".join(reply_parts) + " Can I help you with anything else?"
                
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response answered_product_info_with_context=1")
                return state
        
        # No last recommendations - proceed with normal product_info flow
        if isinstance(tool_result, dict) and tool_result.get("products") and not tool_result.get("product"):
            reply = (
                "We have a few options. To narrow it down, what gender are you looking for (men's, women's, or unisex), "
                "what scent type do you prefer, and what's your budget? "
                "I can recommend one that fits best. Can I help you with anything else?"
            )
            state["messages"].append({"role": "assistant", "content": reply})
            logger.info("node=response ask_clarifying_for_product_info=1")
            return state

    # Handle fallback intent - offer human handoff if bot doesn't understand
    if intent == "fallback":
        reply = "I'm not sure I understand. Would you like to talk with a human agent who can help you better?"
        buttons = [
            {"label": "Yes, connect me", "action": "confirm_handoff", "type": "primary"},
            {"label": "No, thanks", "action": "cancel_handoff", "type": "secondary"},
        ]
        state["button_suggestions"] = buttons
        state["handoff_state"] = "awaiting_confirmation"
        state["messages"].append({"role": "assistant", "content": reply})
        logger.info("node=response fallback_intent offering_handoff=1")
        return state

    tool_context = ""
    if tool_result:
        tool_context = f"Tool result: {tool_result}"
    
    # Build conversation context from recent messages for better context awareness
    recent_messages = state.get("messages", [])[-8:]  # Last 4 exchanges (user + assistant)
    conversation_context = ""
    if len(recent_messages) > 1:  # If there's history beyond current message
        conversation_context = "Recent conversation:\n"
        for msg in recent_messages[:-1]:  # Exclude the current user message (already in messages)
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:200]  # Truncate very long messages
            conversation_context += f"{role}: {content}\n"
    
    # Check if user is responding after a completed action (e.g., handoff completion)
    context_hints = ""
    if state.get("handoff_completed"):
        context_hints = "\nNOTE: You just completed creating a support ticket for the user. They received confirmation."
    
    # Detect if user is expressing gratitude/acknowledgment
    gratitude_words = ["thank", "thanks", "thank you", "appreciate", "great", "perfect", "awesome", "good"]
    user_text_lower = user_text.lower()
    is_gratitude = any(word in user_text_lower for word in gratitude_words)

    final_prompt = f"""{ASSISTANT_STYLE_PROMPT}

{conversation_context}
{context_hints}

Current user message: {user_text}
Detected intent: {intent}
{tool_context}

CRITICAL INSTRUCTIONS:
- Be highly context-aware: Look at the recent conversation to understand what just happened
- If user is thanking you or acknowledging something you did (like creating a ticket), respond warmly and naturally
- Example: If you just created a ticket and user says "thank you", respond like "You're welcome! Happy to help. Is there anything else I can assist you with?"
- If user is just making small talk (like "thank you" after any completed task), acknowledge it briefly and offer more help
- Keep responses very short (1-2 sentences maximum) and conversational
- DO NOT use Markdown, asterisks (**), bold, or lists - plain text only
- Always end naturally by offering further assistance

Write a friendly, context-aware, natural response:
"""
    answer = _call_llm(final_prompt)
    logger.info("node=response llm_called=1 answer_len=%d intent=%s is_gratitude=%s", 
                len(answer) if isinstance(answer, str) else -1, intent, is_gratitude)

    state["messages"].append({"role": "assistant", "content": answer})
    return state



graph = StateGraph(ChatState)
graph.set_entry_point("intent_router")
graph.add_node("intent_router", intent_router)
graph.add_node("auth_guard", auth_guard)
graph.add_node("collect_preferences", collect_preferences)
graph.add_node("tool_exec", tool_exec)
graph.add_node("response_node", response_node)

graph.add_edge("intent_router", "auth_guard")
graph.add_edge("auth_guard", "collect_preferences")
graph.add_edge("collect_preferences", "tool_exec")
graph.add_edge("tool_exec", "response_node")
graph.add_edge("response_node", END)

compiled_app = graph.compile()

