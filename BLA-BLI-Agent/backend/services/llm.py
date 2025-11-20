import os
from typing import List, Dict
from openai import OpenAI


class LLMService:
    """Service for LLM-based chat responses"""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            self.client = OpenAI(api_key=api_key)
            self.enabled = True
        else:
            self.client = None
            self.enabled = False
            print("⚠️  OpenAI API key not found. LLM responses will be simulated.")
    
    def is_product_query(self, message: str) -> bool:
        """
        Determine if the user message is asking for product recommendations
        
        Args:
            message: User message
            
        Returns:
            True if it's a product query, False otherwise
        """
        # Keyword-based detection for product queries
        product_keywords = [
            "recommend", "suggestion", "looking for", "need", "want",
            "buy", "purchase", "shop", "product", "item",
            "show me", "find", "search", "get me",
            "perfume", "cologne", "fragrance", "scent", "smell",
            "men", "women", "unisex", "oud", "trial", "set",
            "spicy", "fresh", "floral", "woody", "sweet", "bold",
            "sandalwood", "vanilla", "cedar", "rose", "jasmine", "amber",
            "dry skin", "skin", "preference", "like", "prefer", "enjoy",
            "cardamom", "tobacco", "bourbon", "bergamot", "lemon", "apple",
            "chamomile", "mandarin", "musk", "lily", "truffle", "berries",
            "lilac", "cinnamon", "nutmeg", "dates", "passionfruit", "caramel",
            "geranium", "patchouli", "honey", "saffron", "tuberose"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in product_keywords)
    
    def analyze_query_intent(self, message: str, history: List[Dict] = None) -> str:
        """
        Analyze the user's message AND history to determine the intent:
        - 'ask_questions': Missing critical info (Gender OR Scent Type).
        - 'show_products': Have Gender AND Scent Type.
        - 'chat': General conversation.
        """
        if not self.enabled:
            # Fallback: simple keyword matching
            if self.is_product_query(message):
                return 'show_products'
            return 'chat'

        try:
            # Format history for context
            history_text = ""
            if history:
                history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

            system_prompt = """You are a shopping assistant for a luxury perfume brand. 
            Your goal is to recommend the PERFECT perfume.
            
            To make a recommendation, you MUST know TWO things:
            1. GENDER (Men, Women, or Unisex)
            2. SCENT PREFERENCE (e.g., Fresh, Floral, Woody, specific notes, or occasion)
            
            Analyze the conversation history and the latest message.
            
            Return 'ask_questions' if:
            - You are missing GENDER.
            - You are missing SCENT PREFERENCE.
            - You need to clarify conflicting info.
            
            Return 'show_products' if:
            - You have BOTH Gender AND Scent Preference (either from the latest message or previous history).
            - The user explicitly asks for a specific product by name.
            
            Return 'chat' if:
            - The user is just greeting or asking general questions (shipping, brand info) unrelated to a specific search.
            
            Return ONLY the string: 'ask_questions', 'show_products', or 'chat'.
            """
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add simplified history context
            if history_text:
                messages.append({"role": "user", "content": f"Previous Conversation:\n{history_text}"})
                
            messages.append({"role": "user", "content": f"Latest Message: {message}"})
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1,
                max_tokens=10
            )
            
            intent = response.choices[0].message.content.strip().lower()
            if intent not in ['ask_questions', 'show_products', 'chat']:
                return 'chat'
            return intent
            
        except Exception as e:
            print(f"❌ Error analyzing intent: {e}")
            return 'chat'

    def generate_response(self, message: str, context: str = "", history: List[Dict] = None) -> str:
        """
        Generate a text response using LLM with history context
        """
        if not self.enabled:
            return self._generate_fallback_response(message)
        
        try:
            system_prompt = """You are a professional shopping assistant for blabliblulife.com.
            
            CRITICAL RULES:
            1. Keep responses SHORT and CONCISE (max 2 sentences).
            2. Ask questions NATURALLY and PROFESSIONALLY. 
               - BAD: "Specify GENDER (Men/Women)"
               - GOOD: "Are you looking for a men's, women's, or unisex fragrance?"
            3. DO NOT repeat questions if the user already answered them.
            4. NEVER invent products.
            
            Context:
            The user is chatting with you. Use the history to understand what they want.
            """
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add history
            if history:
                for msg in history[-5:]: # Keep it to last 5 messages to avoid context limit
                    messages.append({"role": msg["role"], "content": msg["content"]})
            
            if context:
                messages.append({"role": "system", "content": f"System Instruction: {context}"})
                
            messages.append({"role": "user", "content": message})
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"❌ Error generating LLM response: {e}")
            return self._generate_fallback_response(message)

    def generate_product_intro(self, query: str, num_products: int) -> str:
        """
        Generate an introduction message for product recommendations
        
        Args:
            query: User's search query
            num_products: Number of products found
            
        Returns:
            Introduction message
        """
        if num_products == 0:
            return "I couldn't find any products matching your request. Could you try rephrasing or being more specific?"
        
        if not self.enabled:
            return f"Here are {num_products} products I found for you:"
        
        try:
            system_prompt = """You are a professional shopping assistant for blabliblulife.com perfume brand. 
            Write brief, professional introductions for product recommendations."""
            
            prompt = f"Write a brief, professional one-sentence introduction for showing {num_products} perfume recommendations based on the query: '{query}'. Be helpful and professional."
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=60
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"❌ Error generating product intro: {e}")
            return f"Here are {num_products} products I found for you:"


# Global instance
_llm_service = None


def get_llm_service() -> LLMService:
    """Get or create the global LLM service instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
