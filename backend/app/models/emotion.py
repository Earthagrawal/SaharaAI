"""Emotion detection models and data structures."""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np

class EmotionResult(BaseModel):
    """Result from emotion detection."""
    
    valence: float = Field(..., ge=-1.0, le=1.0, description="Emotion valence (-1 to 1)")
    arousal: float = Field(..., ge=-1.0, le=1.0, description="Emotion arousal (-1 to 1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    dominant_emotion: str = Field(..., description="Primary detected emotion")
    emotion_scores: Dict[str, float] = Field(default_factory=dict, description="Scores for each emotion category")
    modality: str = Field(..., description="Detection modality (audio/video)")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class FusedEmotion(BaseModel):
    """Fused emotion result from multiple modalities."""
    
    voice_emotion: Optional[EmotionResult] = None
    face_emotion: Optional[EmotionResult] = None
    fused_valence: float = Field(..., description="Weighted combination of valences")
    fused_arousal: float = Field(..., description="Weighted combination of arousal")
    confidence: float = Field(..., description="Overall confidence")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

def fuse_emotions(voice_result: Optional[EmotionResult], 
                 face_result: Optional[EmotionResult],
                 voice_weight: float = 0.6) -> FusedEmotion:
    """
    Fuse emotion results from voice and face modalities.
    
    Args:
        voice_result: Audio emotion detection result
        face_result: Video emotion detection result  
        voice_weight: Weight for voice emotion (face gets 1 - voice_weight)
    
    Returns:
        Fused emotion result
    """
    face_weight = 1.0 - voice_weight
    
    if voice_result and face_result:
        # Both modalities available
        fused_valence = (voice_weight * voice_result.valence + 
                        face_weight * face_result.valence)
        fused_arousal = (voice_weight * voice_result.arousal + 
                        face_weight * face_result.arousal)
        confidence = (voice_result.confidence + face_result.confidence) / 2
    elif voice_result:
        # Only voice available
        fused_valence = voice_result.valence
        fused_arousal = voice_result.arousal
        confidence = voice_result.confidence * 0.7  # Reduced confidence
    elif face_result:
        # Only face available
        fused_valence = face_result.valence
        fused_arousal = face_result.arousal
        confidence = face_result.confidence * 0.7
    else:
        # No emotions detected
        fused_valence = 0.0
        fused_arousal = 0.0
        confidence = 0.0
    
    return FusedEmotion(
        voice_emotion=voice_result,
        face_emotion=face_result,
        fused_valence=fused_valence,
        fused_arousal=fused_arousal,
        confidence=confidence
    )

def is_sustained_negative(emotion_window: List[FusedEmotion], 
                         threshold: float = -0.35, 
                         sustain_secs: int = 12) -> bool:
    """
    Check if negative emotion is sustained over time window.
    
    Args:
        emotion_window: List of recent emotion results
        threshold: Negative valence threshold
        sustain_secs: Minimum duration in seconds
    
    Returns:
        True if sustained negative emotion detected
    """
    if len(emotion_window) < 2:
        return False
    
    # Check if recent emotions are consistently negative
    negative_count = 0
    total_duration = 0
    
    for emotion in emotion_window:
        if emotion.fused_valence <= threshold:
            negative_count += 1
        
        # Estimate duration (assuming roughly 1 sample per second)
        total_duration += 1
    
    # Check if majority of samples are negative and duration is sufficient
    negative_ratio = negative_count / len(emotion_window)
    return negative_ratio >= 0.7 and total_duration >= sustain_secs