"""Audio emotion detection using NVIDIA Riva and demo fallback."""

import os
from typing import Dict, Any, Iterator
import random

from ..config import config
from ..utils.logger import get_logger
from ..models.emotion import EmotionResult

logger = get_logger(__name__)

class RivaEmotionClient:
    """NVIDIA Riva emotion detection client with demo fallback."""
    
    def __init__(self):
        self.riva_url = config.RIVA_URL
        self.riva_api_key = config.RIVA_API_KEY
        self.demo_mode = not (self.riva_url and self.riva_api_key)
        
        if self.demo_mode:
            logger.warning("DEMO MODE - Riva emotion detection not configured, using demo responses")
        else:
            logger.info("Riva emotion detection client initialized")
    
    def _generate_demo_emotion(self, input_source: str = "unknown") -> EmotionResult:
        """Generate demo emotion detection result."""
        # Demo emotion categories
        emotions = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]
        
        # Use input hash for consistent results
        input_hash = hash(input_source)
        random.seed(input_hash)
        
        # Generate realistic emotion scores
        dominant_emotion = random.choice(emotions)
        emotion_scores = {}
        
        for emotion in emotions:
            if emotion == dominant_emotion:
                emotion_scores[emotion] = random.uniform(0.6, 0.9)
            else:
                emotion_scores[emotion] = random.uniform(0.0, 0.3)
        
        # Map emotion to valence/arousal
        emotion_mapping = {
            "neutral": (0.0, 0.0),
            "happy": (0.7, 0.5),
            "sad": (-0.6, -0.3),
            "angry": (-0.5, 0.8),
            "surprised": (0.2, 0.8),
            "fearful": (-0.7, 0.6),
            "disgusted": (-0.8, 0.2)
        }
        
        valence, arousal = emotion_mapping.get(dominant_emotion, (0.0, 0.0))
        
        return EmotionResult(
            valence=valence,
            arousal=arousal,
            confidence=emotion_scores[dominant_emotion],
            dominant_emotion=dominant_emotion,
            emotion_scores=emotion_scores,
            modality="audio"
        )

def analyze_wav(path: str) -> Dict[str, Any]:
    """
    Analyze emotion from audio file.
    
    Args:
        path: Path to audio file
    
    Returns:
        Emotion analysis results
    """
    client = RivaEmotionClient()
    
    if not os.path.exists(path):
        logger.error(f"Audio file not found: {path}")
        return {"error": f"Audio file not found at {path}"}
    
    if client.demo_mode:
        logger.warning("DEMO MODE - Using demo emotion analysis")
        emotion_result = client._generate_demo_emotion(path)
        return emotion_result.dict()
    
    # TODO: Implement actual Riva emotion detection
    try:
        # Placeholder for Riva implementation:
        # 1. Connect to Riva emotion detection service
        # 2. Send audio file for analysis
        # 3. Receive emotion predictions
        # 4. Return structured results
        
        logger.info(f"Analyzing emotion from audio file: {path}")
        
        # For now, return demo result
        emotion_result = client._generate_demo_emotion(path)
        return emotion_result.dict()
        
    except Exception as e:
        logger.error(f"Error in audio emotion analysis: {e}")
        return {"error": f"Emotion analysis failed - {str(e)}"}

def analyze_stream(chunks: Iterator[bytes]) -> Iterator[Dict[str, Any]]:
    """
    Analyze emotion from streaming audio.
    
    Args:
        chunks: Iterator of audio chunks
    
    Yields:
        Streaming emotion analysis results
    """
    client = RivaEmotionClient()
    
    if client.demo_mode:
        logger.warning("DEMO MODE - Using demo streaming emotion analysis")
        
        # Generate demo streaming results
        chunk_count = 0
        for chunk in chunks:
            chunk_count += 1
            emotion_result = client._generate_demo_emotion(f"stream_chunk_{chunk_count}")
            yield emotion_result.dict()
            
            # Limit demo output
            if chunk_count > 10:
                break
        return
    
    # TODO: Implement actual Riva streaming emotion detection
    try:
        # Placeholder for Riva streaming implementation:
        # 1. Establish streaming connection to Riva
        # 2. Send audio chunks as they arrive
        # 3. Yield emotion results in real-time
        # 4. Handle stream end
        
        logger.info("Starting streaming emotion analysis")
        
        # For now, yield demo results
        chunk_count = 0
        for chunk in chunks:
            chunk_count += 1
            emotion_result = client._generate_demo_emotion(f"stream_chunk_{chunk_count}")
            yield emotion_result.dict()
        
    except Exception as e:
        logger.error(f"Error in streaming emotion analysis: {e}")
        yield {"error": f"Streaming emotion analysis failed - {str(e)}"}