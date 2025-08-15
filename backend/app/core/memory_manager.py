"""Memory management system for conversation history and context."""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict, deque

from ..config import config
from ..utils.logger import get_logger
from .llm_client import summarize

logger = get_logger(__name__)

class MemoryManager:
    """Manages short-term and long-term memory for conversations."""
    
    def __init__(self, max_short_term: int = 10):
        """
        Initialize memory manager.
        
        Args:
            max_short_term: Maximum number of turns to keep in short-term memory
        """
        self.max_short_term = max_short_term
        self.short_term_memory = defaultdict(lambda: deque(maxlen=max_short_term))
        self.conversation_dir = config.CONVERSATIONS_DIR
        self.memory_summary_path = config.MEMORY_SUMMARY_PATH
        
        # Ensure directories exist
        os.makedirs(self.conversation_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.memory_summary_path), exist_ok=True)
    
    def _get_conversation_path(self, session_id: str) -> str:
        """Get file path for session conversation history."""
        return os.path.join(self.conversation_dir, f"{session_id}.json")
    
    def _load_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Load conversation history from disk."""
        conv_path = self._get_conversation_path(session_id)
        if os.path.exists(conv_path):
            try:
                with open(conv_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading conversation history: {e}")
        return []
    
    def _save_conversation_history(self, session_id: str, history: List[Dict[str, Any]]):
        """Save conversation history to disk."""
        conv_path = self._get_conversation_path(session_id)
        try:
            with open(conv_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving conversation history: {e}")
    
    def _extract_summary(self, turns: List[Dict[str, Any]]) -> str:
        """
        Extract summary from conversation turns using LLM or fallback method.
        
        Args:
            turns: List of conversation turns
        
        Returns:
            Summary string
        """
        if not turns:
            return "No conversation content"
        
        try:
            # Try LLM summarization first
            texts = []
            for turn in turns:
                if turn.get('user_message'):
                    texts.append(f"User: {turn['user_message']}")
                if turn.get('assistant_response'):
                    texts.append(f"Assistant: {turn['assistant_response']}")
            
            if texts:
                summary = summarize(texts)
                if summary and not "Error:" in summary:
                    return summary
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
        
        # Fallback: simple extraction
        return self._simple_summary(turns)
    
    def _simple_summary(self, turns: List[Dict[str, Any]]) -> str:
        """Simple fallback summarization method."""
        if not turns:
            return "Empty conversation"
        
        topics = set()
        key_points = []
        
        for turn in turns:
            # Extract key terms from user messages
            if turn.get('user_message'):
                words = turn['user_message'].lower().split()
                # Simple keyword extraction (words longer than 4 characters)
                for word in words:
                    if len(word) > 4 and word.isalpha():
                        topics.add(word)
            
            # Extract emotion info if available
            if turn.get('emotion_summary'):
                key_points.append(f"Emotion: {turn['emotion_summary']}")
        
        summary_parts = []
        if topics:
            summary_parts.append(f"Topics: {', '.join(list(topics)[:5])}")
        if key_points:
            summary_parts.append("; ".join(key_points[:3]))
        
        return "; ".join(summary_parts) if summary_parts else "General conversation"

def append_turn(session_id: str, turn: Dict[str, Any]) -> None:
    """
    Append a conversation turn to memory.
    
    Args:
        session_id: Session identifier
        turn: Turn data with user_message, assistant_response, timestamp, etc.
    """
    memory = MemoryManager()
    
    # Add timestamp if not present
    if 'timestamp' not in turn:
        turn['timestamp'] = datetime.now().isoformat()
    
    # Add to short-term memory
    memory.short_term_memory[session_id].append(turn)
    
    # Load and update persistent history
    history = memory._load_conversation_history(session_id)
    history.append(turn)
    memory._save_conversation_history(session_id, history)
    
    logger.info(f"Added turn to session {session_id}")

def get_recent(session_id: str, k: int) -> List[Dict[str, Any]]:
    """
    Get recent conversation turns.
    
    Args:
        session_id: Session identifier
        k: Number of recent turns to retrieve
    
    Returns:
        List of recent turns
    """
    memory = MemoryManager()
    
    # First check short-term memory
    short_term = list(memory.short_term_memory[session_id])
    if len(short_term) >= k:
        return short_term[-k:]
    
    # Load from disk if needed
    history = memory._load_conversation_history(session_id)
    return history[-k:] if history else []

def persist_session(session_id: str) -> None:
    """
    Persist session to long-term memory summary.
    
    Args:
        session_id: Session identifier to persist
    """
    memory = MemoryManager()
    
    # Load full conversation history
    history = memory._load_conversation_history(session_id)
    
    if not history:
        logger.warning(f"No history found for session {session_id}")
        return
    
    # Generate summary
    summary = memory._extract_summary(history)
    
    # Extract tags/topics
    tags = _extract_tags(history)
    
    # Create summary entry
    timestamp = datetime.now().strftime("%Y-%m-%d")
    summary_line = f"{timestamp}|{','.join(tags)}|{summary}\n"
    
    # Append to memory summary file
    try:
        with open(memory.memory_summary_path, 'a', encoding='utf-8') as f:
            f.write(summary_line)
        logger.info(f"Persisted session {session_id} to long-term memory")
    except Exception as e:
        logger.error(f"Error persisting session: {e}")
    
    # Clear short-term memory for this session
    if session_id in memory.short_term_memory:
        memory.short_term_memory[session_id].clear()

def _extract_tags(history: List[Dict[str, Any]]) -> List[str]:
    """Extract tags/topics from conversation history."""
    tags = set()
    
    for turn in history:
        # Extract from emotion data
        if turn.get('emotion_summary'):
            emotion = turn['emotion_summary'].lower()
            if 'negative' in emotion or 'distress' in emotion:
                tags.add('emotional')
            if 'positive' in emotion:
                tags.add('positive')
        
        # Extract from user messages (simple keyword extraction)
        if turn.get('user_message'):
            message = turn['user_message'].lower()
            if any(word in message for word in ['help', 'problem', 'issue']):
                tags.add('support')
            if any(word in message for word in ['question', 'how', 'what', 'why']):
                tags.add('inquiry')
    
    return list(tags)[:3]  # Limit to 3 tags