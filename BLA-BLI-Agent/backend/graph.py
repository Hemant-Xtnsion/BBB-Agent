import os
from typing import TypedDict, List, Optional, Dict, Any
import re
import logging

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
    
    # Check if user is asking about previously recommended products
    # This helps improve intent detection for product_question
    has_last_recs = bool(state.get("last_recommendations"))
    
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
    if intent not in ("product_info", "product_recommendation"):
        return state

    # Get all user messages from conversation history to extract preferences from context
    user_messages = [m["content"] for m in state["messages"] if m["role"] == "user"]
    last_user_msg = user_messages[-1] if user_messages else ""
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
    }

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
        
        # Merge extracted prefs into existing prefs
        if extracted_prefs.get("gender"):
            prefs["gender"] = extracted_prefs["gender"]
        
        if extracted_prefs.get("scent_types"):
            existing_scents = prefs.get("scent_types", [])
            new_scents = extracted_prefs["scent_types"]
            prefs["scent_types"] = list(set(existing_scents + new_scents))
            
        if extracted_prefs.get("budget"):
            prefs["budget"] = extracted_prefs["budget"]
            
        if extracted_prefs.get("is_gift") is not None:
            # Only update is_gift if explicitly detected
            # But respect the "single perfume" override from recommender if present
            # We'll let recommender logic handle the fine-grained is_gift logic
            # just set it here if true
            if extracted_prefs["is_gift"]:
                prefs["is_gift"] = True
            elif extracted_prefs["is_gift"] is False:
                 # If LLM explicitly says not a gift, we might want to trust it
                 # But let's be careful not to override previous "true" unless sure
                 pass

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
            # For gift queries, only need gender. For regular queries, need gender AND (scent_type OR budget)
            is_gift = new_prefs.get("is_gift", False)
            if is_gift:
                has_sufficient_prefs = new_prefs.get("gender") is not None
            else:
                has_sufficient_prefs = (
                    new_prefs.get("gender") is not None and
                    (
                        (isinstance(new_prefs.get("scent_types"), list) and len(new_prefs.get("scent_types", [])) > 0) or
                        new_prefs.get("budget") is not None
                    )
                )
            
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
    elif intent == "product_recommendation":
        # If collect_preferences already set recommendations (with sufficient prefs), use them
        if state.get("tool_result") and isinstance(state.get("tool_result"), dict) and "recommendations" in state["tool_result"]:
            logger.info("node=tool_exec path=product_recommendation using_pre_filled_recommendations")
            # Reset recommendation index when new recommendations are generated
            state["recommendation_index"] = 0
            return state
        
        # If collect_preferences set tool_result to None (insufficient prefs), preserve it
        if state.get("tool_result") is None:
            logger.info("node=tool_exec path=product_recommendation preserving_none_from_collect_prefs")
            return state
        
        # Check if we have sufficient preferences before recommending
        # Require gender AND at least one other preference (scent_type OR budget)
        prefs = state.get("preferences") or {}
        has_gender = prefs.get("gender") is not None
        has_scent_type = isinstance(prefs.get("scent_types"), list) and len(prefs.get("scent_types", [])) > 0
        has_budget = prefs.get("budget") is not None
        has_sufficient_prefs = has_gender and (has_scent_type or has_budget)
        
        # Only recommend if we have sufficient preferences
        if has_sufficient_prefs:
            user_text = next(m["content"] for m in reversed(state["messages"]) if m["role"] == "user")
            recs = recommend_products(user_text)
            result = {"recommendations": recs}
            # Reset recommendation index when new recommendations are generated
            state["recommendation_index"] = 0
            logger.info("node=tool_exec path=product_recommendation count=%d", len(recs))
        else:
            # No sufficient preferences - don't set recommendations, response_node will ask questions
            result = None
            logger.info("node=tool_exec path=product_recommendation insufficient_prefs gender=%s scent=%s budget=%s", 
                       has_gender, has_scent_type, has_budget)
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
            
            # Try to find product by name if mentioned
            product = None
            for rec in last_recs:
                rec_name = rec.get("name", "").lower()
                # Check if product name is mentioned in user text
                if rec_name and any(word in user_text_lower for word in rec_name.split()):
                    product = rec
                    break
            
            # If no specific product found, use first one
            if not product:
                product = last_recs[0] if last_recs else {}
            
            # Handle price questions
            if any(word in user_text_lower for word in ["price", "cost", "how much"]):
                price = product.get("price", 0)
                name = product.get("name", "this product")
                if price:
                    reply = f"The price of {name} is ₹{price}. Can I help you with anything else?"
                else:
                    reply = f"I don't have the price information for {name} right now. Can I help you with anything else?"
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response answered_price_question=1")
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
                
                reply = ". ".join(reply_parts) + ". Can I help you with anything else?"
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response answered_contents_question=1")
                return state
            
            # Handle other questions - use LLM but with structured product data
            import json
            products_context = json.dumps(last_recs[:2], indent=2)
            prompt = f"""{ASSISTANT_STYLE_PROMPT}

The user is asking about these recommended products:
{products_context}

User question: {user_text}

IMPORTANT: Extract information directly from the product data above. Do not make up prices or details.
- Price is in the "price" field (in USD)
- Description is in "short_description"
- Notes are in "notes" object with "head", "heart", "base" arrays
- Name is in "name" field

Answer their question naturally and conversationally using ONLY the data provided above.
Keep it short and friendly. End with: "Can I help you with anything else?"
"""
            answer = _call_llm(prompt)
            state["messages"].append({"role": "assistant", "content": answer})
            logger.info("node=response answered_product_question=1")
            return state
        else:
            # No last recommendations - might be asking about a product in general
            reply = "I don't have any products in context right now. Could you tell me what you're looking for? Can I help you with anything else?"
            state["messages"].append({"role": "assistant", "content": reply})
            logger.info("node=response product_question_no_context=1")
            return state

    # For recommendations: ask preferences one by one, remember from context
    if intent == "product_recommendation":
        # Check if we have sufficient preferences before recommending
        prefs = state.get("preferences") or {}
        has_gender = prefs.get("gender") is not None
        has_scent_type = isinstance(prefs.get("scent_types"), list) and len(prefs.get("scent_types", [])) > 0
        has_budget = prefs.get("budget") is not None
        is_gift = prefs.get("is_gift", False)
        
        # For gift queries, we can recommend with just gender (gift sets are versatile)
        # For regular queries, require gender AND (scent_type OR budget)
        if is_gift:
            has_sufficient_prefs = has_gender  # Gifts only need gender
        else:
            has_sufficient_prefs = has_gender and (has_scent_type or has_budget)
        
        # Only recommend if we have sufficient preferences AND recommendations exist
        if has_sufficient_prefs:
            if isinstance(tool_result, dict) and tool_result.get("recommendations"):
                recs = tool_result["recommendations"]
                
                # Check if user is asking for "other options" or "anything else"
                user_text_lower = user_text.lower()
                asking_for_other = any(phrase in user_text_lower for phrase in [
                    "other option", "another option", "other options", "anything else", "something else",
                    "show me another", "different one", "what else", "more options"
                ])
                
                # Get current recommendation index (default to 0 if not set)
                current_index = state.get("recommendation_index", 0)
                
                # If asking for other options, increment index
                if asking_for_other:
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
                    
                    # Build a natural response with positioning line
                    reply_parts = [f"I'd suggest {name}."]
                    if positioning:
                        reply_parts.append(positioning)
                    if desc:
                        reply_parts.append(desc)
                    if price:
                        reply_parts.append(f"Priced at ₹{price}.")
                    
                    # Customize ending based on how many options are left
                    remaining = len(recs) - current_index - 1
                    if remaining > 0:
                        reply_parts.append("Want to know more or see other options?")
                    else:
                        reply_parts.append("This is the best match for your preferences. Want to know more?")
                    
                    reply = " ".join(reply_parts)
                    
                    state["messages"].append({"role": "assistant", "content": reply})
                    logger.info("node=response used_recommendations=1 showing_product_index=%d remaining=%d", current_index, remaining)
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
                reply = "I couldn't find any perfumes matching your exact criteria."
                if has_budget:
                    reply += " It might be the budget constraint. Would you like to see options with a slightly higher budget?"
                else:
                    reply += " Would you like to adjust your preferences?"
                
                state["messages"].append({"role": "assistant", "content": reply})
                logger.info("node=response no_recommendations_found sufficient_prefs=1")
                return state
        
        # Ask for missing preferences ONE AT A TIME
        # Priority: gender > scent_type > budget
        if is_gift:
            if not has_gender:
                reply = "Is this gift for men, women, or would you prefer a unisex option?"
            else:
                # Should have recommendations if gender is set for gifts
                reply = "Let me find the perfect gift set for you."
        else:
            # Regular product recommendation flow - ask ONE question at a time
            if not has_gender:
                reply = "Are you looking for men's, women's, or unisex perfume?"
            elif not has_scent_type:
                # Have gender, ask for scent type next
                reply = "What scent type do you prefer - fresh, floral, woody, oud, or sweet?"
            elif not has_budget:
                # Have gender and scent type, ask for budget (optional but helpful)
                reply = "What's your budget range?"
            else:
                # Shouldn't reach here if has_sufficient_prefs logic is correct
                # But if we do, it means we have prefs but maybe tool_result was missing/error
                reply = "Let me find the perfect perfume for you."
        
        state["messages"].append({"role": "assistant", "content": reply})
        logger.info("node=response ask_clarifying_for_reco=1 missing_gender=%s missing_scent=%s missing_budget=%s is_gift=%s", 
                   not has_gender, not has_scent_type, not has_budget, is_gift)
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

