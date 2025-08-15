"""Memory models and data structures."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class ConversationTurn(BaseModel):
    """Single conversation turn."""
    
    turn_id: str = Field(..., description="Unique turn identifier")
    user_message: Optional[str] = Field(default=None, description="User message")
    assistant_response: Optional[str] = Field(default=None, description="Assistant response")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    emotion_summary: Optional[str] = Field(default=None, description="Emotion state summary")
    context_used: List[str] = Field(default_factory=list, description="Knowledge sources used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class MemorySummary(BaseModel):
    """Summary of conversation memory."""
    
    session_id: str = Field(..., description="Session identifier")
    date: str = Field(..., description="Date of conversation")
    tags: List[str] = Field(default_factory=list, description="Topic tags")
    summary_text: str = Field(..., description="Summary of the conversation")
    turn_count: int = Field(default=0, description="Number of turns in conversation")
    emotion_highlights: List[str] = Field(default_factory=list, description="Notable emotional moments")

class UserProfile(BaseModel):
    """User profile information."""
    
    user_id: str = Field(..., description="User identifier")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    conversation_history_summary: List[MemorySummary] = Field(default_factory=list, description="Historical conversation summaries")
    emotional_patterns: Dict[str, Any] = Field(default_factory=dict, description="Observed emotional patterns")
    interaction_statistics: Dict[str, Any] = Field(default_factory=dict, description="Usage statistics")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class KnowledgeBase(BaseModel):
    """Knowledge base document."""
    
    doc_id: str = Field(..., description="Document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    source: str = Field(..., description="Source of the document")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ContextWindow(BaseModel):
    """Context window for conversation."""
    
    recent_turns: List[ConversationTurn] = Field(default_factory=list, description="Recent conversation turns")
    memory_summary: Optional[str] = Field(default=None, description="Summarized memory context")
    user_profile_snippet: Optional[Dict[str, Any]] = Field(default=None, description="Relevant user profile information")
    knowledge_context: List[Dict[str, Any]] = Field(default_factory=list, description="Relevant knowledge base excerpts")
    emotion_context: Optional[Dict[str, Any]] = Field(default=None, description="Current emotional context")
    system_instructions: Optional[str] = Field(default=None, description="System-level instructions")