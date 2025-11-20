from fastapi import APIRouter, HTTPException, Response
from models.product import ChatMessage, ChatResponse, Product
from models.chat import ChatRequest as GraphChatRequest, ChatResponse as GraphChatResponse, Message
from services.rag import get_rag_service
from services.llm import get_llm_service
from services.history import get_history_service
from graph import compiled_app
from tools import INIT_PUBLIC_DATA
from typing import List, Optional
import logging

logger = logging.getLogger("blabli.api")
router = APIRouter()


@router.options("/chat")
async def chat_options():
    """
    Explicit OPTIONS handler so browser preflight checks succeed even before POST.
    """
    return Response(status_code=200)


@router.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """
    Handle chat messages using LangGraph flow with order tracking support.
    Falls back to RAG for product search when appropriate.
    """
    try:
        user_message = message.message.strip()
        conversation_id = message.conversation_id or "default"
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Convert to graph format - build messages from history
        history_service = get_history_service()
        history = history_service.get_history(conversation_id)
        
        # Build messages list for graph
        messages = []
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Get email and order_number from request or extract from message
        email = message.email
        order_number = message.order_number
        
        # If not provided in request, try to extract from message
        if not email:
            import re
            email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", user_message)
            if email_match:
                email = email_match.group(0)
        
        if not order_number:
            import re
            order_match = re.search(r"\bBLB-\d{3,}\b", user_message, flags=re.IGNORECASE)
            if order_match:
                order_number = order_match.group(0).upper()
        
        graph_request = GraphChatRequest(
            messages=[Message(role=m["role"], content=m["content"]) for m in messages],
            email=email,
            order_number=order_number,
            conversation_id=conversation_id
        )
        
        # Load persisted state (preferences, last_recommendations, etc.)
        persisted_state = history_service.get_state(conversation_id)
        
        # Run graph
        initial_state = {
            "messages": [m.dict() for m in graph_request.messages],
            "customer_authed": False,
            "customer_email": graph_request.email,
            "order_number": graph_request.order_number,
            "preferences": persisted_state.get("preferences"),
            "last_recommendations": persisted_state.get("last_recommendations"),
            "recommendation_index": persisted_state.get("recommendation_index", 0),
        }
        
        result_state = compiled_app.invoke(initial_state)
        
        # Persist state after graph execution
        if result_state.get("preferences"):
            history_service.update_state(conversation_id, preferences=result_state["preferences"])
        if result_state.get("last_recommendations"):
            history_service.update_state(conversation_id, last_recommendations=result_state["last_recommendations"])
        if "recommendation_index" in result_state:
            history_service.update_state(conversation_id, recommendation_index=result_state["recommendation_index"])
        
        # Extract assistant response
        assistant_msgs = [m for m in result_state["messages"] if m["role"] == "assistant"]
        last_assistant = assistant_msgs[-1]["content"] if assistant_msgs else "I'm here."
        
        # Extract product recommendations if available
        products = None
        tool_result = result_state.get("tool_result")
        if tool_result and isinstance(tool_result, dict):
            recommendations = tool_result.get("recommendations")
            if recommendations and isinstance(recommendations, list) and len(recommendations) > 0:
                # Get the current recommendation index from state
                current_index = result_state.get("recommendation_index", 0)
                
                # Show the product at the current index
                if current_index < len(recommendations):
                    products_to_show = [recommendations[current_index]]
                else:
                    # Fallback to first product if index is out of range
                    products_to_show = [recommendations[0]]
                
                # Convert to Product models for frontend
                products = []
                for rec in products_to_show:
                    # Handle both graph format and RAG format
                    # Map image_url to thumbnail if thumbnail is not present
                    thumbnail = rec.get("thumbnail") or rec.get("image_url", "")
                    product_url = rec.get("url") or rec.get("product_url", "")
                    description = rec.get("description") or rec.get("short_description", "")
                    
                    product = {
                        "title": rec.get("name") or rec.get("title", ""),
                        "price": rec.get("price") or rec.get("price_usd", "") or "",
                        "description": description,
                        "thumbnail": thumbnail,
                        "url": product_url,
                        "tags": rec.get("tags", []),
                        "category": rec.get("category", ""),
                    }
                    products.append(product)
        
        # Update history
        history_service.add_message(conversation_id, "user", user_message)
        history_service.add_message(conversation_id, "assistant", last_assistant)
        
        # Determine response type
        response_type = "products" if products else "text"
        
        # Convert products to Product models if present
        product_models = []
        if products:
            product_models = [
                Product(
                    title=p.get("title", ""),
                    price=str(p.get("price", "")),
                    thumbnail=p.get("thumbnail", ""),
                    url=p.get("url", ""),
                    description=p.get("description", ""),
                    tags=p.get("tags", []),
                    category=p.get("category", "")
                )
                for p in products
            ]
        
        # Extract state from graph result
        response_state = {
            "customer_authed": result_state.get("customer_authed", False),
            "customer_email": result_state.get("customer_email"),
            "order_number": result_state.get("order_number"),
            "intent": result_state.get("intent"),
            "missing_field": result_state.get("missing_field"),
        }
        
        return ChatResponse(
            type=response_type,
            message=last_assistant,
            products=product_models,
            state=response_state
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in chat endpoint")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.options("/chat/graph")
async def chat_graph_options():
    return Response(status_code=200)


@router.post("/chat/graph", response_model=GraphChatResponse)
async def chat_graph(req: GraphChatRequest):
    """
    Direct graph endpoint for advanced features (order tracking, etc.)
    Uses the full LangGraph flow with state management.
    """
    try:
        logger.info(
            "endpoint=/chat/graph received messages=%d has_email=%s has_order=%s",
            len(req.messages),
            bool(req.email),
            bool(req.order_number),
        )
        
        # LangGraph expects a state-like dict
        initial_state = {
            "messages": [m.dict() for m in req.messages],
            "customer_authed": False,
            "customer_email": req.email,
            "order_number": req.order_number,
        }

        # run one turn
        result_state = compiled_app.invoke(initial_state)

        logger.info(
            "endpoint=/chat/graph result intent=%s authed=%s missing=%s has_tool_result=%s",
            result_state.get("intent"),
            result_state.get("customer_authed"),
            result_state.get("missing_field"),
            bool(result_state.get("tool_result")),
        )

        # last assistant msg
        assistant_msgs = [m for m in result_state["messages"] if m["role"] == "assistant"]
        last_assistant = assistant_msgs[-1]["content"] if assistant_msgs else "I'm here."

        # Extract product recommendations if available
        products = None
        tool_result = result_state.get("tool_result")
        if tool_result and isinstance(tool_result, dict):
            recommendations = tool_result.get("recommendations")
            if recommendations and isinstance(recommendations, list) and len(recommendations) > 0:
                # Map recommendations to products array format
                products = []
                for rec in recommendations:
                    # Map image_url to thumbnail if thumbnail is not present
                    thumbnail = rec.get("thumbnail") or rec.get("image_url", "")
                    product_url = rec.get("url") or rec.get("product_url", "")
                    description = rec.get("description") or rec.get("short_description", "")
                    
                    product = {
                        "id": rec.get("id", ""),
                        "name": rec.get("name") or rec.get("title", ""),
                        "price": rec.get("price") or rec.get("price_usd", 0),
                        "description": description,
                        "thumbnail": thumbnail,
                        "url": product_url,
                    }
                    # Include additional fields if present
                    if "gender" in rec:
                        product["gender"] = rec["gender"]
                    if "scent_type" in rec:
                        product["scent_type"] = rec["scent_type"]
                    if "score" in rec:
                        product["score"] = rec["score"]
                    products.append(product)

        return GraphChatResponse(
            reply=last_assistant,
            state={
                "customer_authed": result_state.get("customer_authed", False),
                "customer_email": result_state.get("customer_email"),
                "order_number": result_state.get("order_number"),
                "intent": result_state.get("intent"),
            },
            products=products,
        )
    except Exception as e:
        logger.exception("Error in graph chat endpoint")
        raise HTTPException(status_code=500, detail=str(e))
