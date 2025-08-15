"""Chat models and data structures."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class ChatMessage(BaseModel):
    """Individual chat message."""
    
    role: str = Field(..., description="Message role (user/assistant/system)")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    emotion_data: Optional[Dict[str, Any]] = Field(default=None, description="Associated emotion data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ChatSession(BaseModel):
    """Chat session containing multiple messages."""
    
    session_id: str = Field(..., description="Unique session identifier")
    messages: List[ChatMessage] = Field(default_factory=list, description="List of messages")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    user_profile: Optional[Dict[str, Any]] = Field(default=None, description="User profile data")
    context: Dict[str, Any] = Field(default_factory=dict, description="Session context")

class ChatRequest(BaseModel):
    """Request for chat completion."""
    
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    include_context: bool = Field(default=True, description="Include conversation context")
    emotion_context: Optional[Dict[str, Any]] = Field(default=None, description="Current emotion state")

class ChatResponse(BaseModel):
    """Response from chat completion."""
    
    response: str = Field(..., description="Assistant response")
    session_id: str = Field(..., description="Session identifier")
    distress_detected: bool = Field(default=False, description="Whether distress was detected")
    helpline_info: Optional[Dict[str, Any]] = Field(default=None, description="Helpline information if distress detected")
    emotion_summary: Optional[str] = Field(default=None, description="Summary of detected emotions")
    context_used: List[str] = Field(default_factory=list, description="Context sources used")

class DistressAlert(BaseModel):
    """Distress detection alert."""
    
    session_id: str = Field(..., description="Session where distress was detected")
    trigger_type: str = Field(..., description="Type of trigger (text/emotion/sustained)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    message_content: Optional[str] = Field(default=None, description="Triggering message")
    emotion_data: Optional[Dict[str, Any]] = Field(default=None, description="Associated emotion data")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    helpline_provided: bool = Field(default=False, description="Whether helpline info was provided")