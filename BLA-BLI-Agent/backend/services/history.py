from typing import List, Dict, Optional
import time

class HistoryService:
    """
    Simple in-memory chat history service.
    In a production app, this should be replaced with a database (Redis/Postgres).
    """
    
    def __init__(self):
        # Dict[conversation_id, List[Dict[role, content]]]
        self._history: Dict[str, List[Dict[str, str]]] = {}
        # Conversation state (preferences, last_recommendations, etc.)
        self._state: Dict[str, Dict] = {}
        # Cleanup tracking
        self._last_access: Dict[str, float] = {}
        
    def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get conversation history"""
        if not conversation_id:
            return []
        
        self._last_access[conversation_id] = time.time()
        return self._history.get(conversation_id, [])
    
    def add_message(self, conversation_id: str, role: str, content: str):
        """Add a message to history"""
        if not conversation_id:
            return
            
        if conversation_id not in self._history:
            self._history[conversation_id] = []
            
        self._history[conversation_id].append({
            "role": role,
            "content": content
        })
        self._last_access[conversation_id] = time.time()
        
        # Keep history manageable (last 10 messages)
        if len(self._history[conversation_id]) > 10:
            self._history[conversation_id] = self._history[conversation_id][-10:]
    
    def get_state(self, conversation_id: str) -> Dict:
        """Get conversation state (preferences, last_recommendations, etc.)"""
        if not conversation_id:
            return {}
        return self._state.get(conversation_id, {})
    
    def update_state(self, conversation_id: str, **kwargs):
        """Update conversation state"""
        if not conversation_id:
            return
        if conversation_id not in self._state:
            self._state[conversation_id] = {}
        self._state[conversation_id].update(kwargs)
        self._last_access[conversation_id] = time.time()

    def clear_history(self, conversation_id: str):
        """Clear history for a conversation"""
        if conversation_id in self._history:
            del self._history[conversation_id]
        if conversation_id in self._state:
            del self._state[conversation_id]
        if conversation_id in self._last_access:
            del self._last_access[conversation_id]

# Global instance
_history_service = None

def get_history_service() -> HistoryService:
    global _history_service
    if _history_service is None:
        _history_service = HistoryService()
    return _history_service
