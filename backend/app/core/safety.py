"""Safety detection and crisis intervention system."""

import re
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ..config import config
from ..utils.logger import get_logger
from ..models.emotion import FusedEmotion, is_sustained_negative
from ..models.chat import DistressAlert

logger = get_logger(__name__)

class SafetyMonitor:
    """Monitors conversations for signs of distress and crisis situations."""
    
    def __init__(self):
        self.helplines_path = config.HELPLINES_PATH
        self.distress_keywords = []
        self.helpline_data = {}
        self._load_helpline_data()
    
    def _load_helpline_data(self):
        """Load helpline information from JSON file."""
        try:
            if os.path.exists(self.helplines_path):
                with open(self.helplines_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.helpline_data = data
                    self.distress_keywords = data.get('distress_keywords', [])
                logger.info(f"Loaded {len(self.distress_keywords)} distress keywords and helpline data")
            else:
                logger.warning(f"Helpline data file not found: {self.helplines_path}")
                self._create_default_helpline_data()
        except Exception as e:
            logger.error(f"Error loading helpline data: {e}")
            self._create_default_helpline_data()
    
    def _create_default_helpline_data(self):
        """Create default helpline data if file doesn't exist."""
        self.distress_keywords = [
            "suicide", "kill myself", "end it all", "can't go on", "want to die",
            "hopeless", "no point", "give up", "hurt myself", "self harm"
        ]
        self.helpline_data = {
            "global": {
                "name": "Crisis Support",
                "message": "If you're in crisis, please reach out for help immediately."
            }
        }
    
    def detect_text_distress(self, text: str) -> Tuple[bool, float, List[str]]:
        """
        Detect distress indicators in text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Tuple of (is_distressed, confidence, matched_keywords)
        """
        if not text:
            return False, 0.0, []
        
        text_lower = text.lower()
        matched_keywords = []
        
        # Check for direct keyword matches
        for keyword in self.distress_keywords:
            if keyword.lower() in text_lower:
                matched_keywords.append(keyword)
        
        # Additional pattern matching
        crisis_patterns = [
            r'\b(want to|going to|plan to)\s+(die|kill|hurt)\b',
            r'\b(no\s+hope|hopeless|worthless|burden)\b',
            r'\b(can\'?t\s+(go on|continue|take it))\b',
            r'\b(end\s+it\s+all|give\s+up)\b'
        ]
        
        for pattern in crisis_patterns:
            if re.search(pattern, text_lower):
                matched_keywords.append("crisis_pattern")
                break
        
        # Calculate confidence based on number and severity of matches
        if matched_keywords:
            base_confidence = min(len(matched_keywords) * 0.3, 0.9)
            
            # High-risk keywords increase confidence
            high_risk = ["suicide", "kill myself", "want to die", "end it all"]
            if any(keyword in matched_keywords for keyword in high_risk):
                base_confidence = max(base_confidence, 0.8)
            
            return True, base_confidence, matched_keywords
        
        return False, 0.0, []
    
    def detect_emotion_distress(self, emotion_window: List[FusedEmotion]) -> Tuple[bool, float]:
        """
        Detect sustained negative emotions indicating distress.
        
        Args:
            emotion_window: Recent emotion history
        
        Returns:
            Tuple of (is_distressed, confidence)
        """
        if not emotion_window:
            return False, 0.0
        
        try:
            # Check for sustained negative emotion
            is_sustained = is_sustained_negative(emotion_window)
            
            if is_sustained:
                # Calculate confidence based on emotion intensity and duration
                recent_emotions = emotion_window[-5:]  # Last 5 readings
                avg_valence = sum(e.fused_valence for e in recent_emotions) / len(recent_emotions)
                avg_confidence = sum(e.confidence for e in recent_emotions) / len(recent_emotions)
                
                # More negative valence = higher distress confidence
                distress_confidence = abs(avg_valence) * avg_confidence
                return True, min(distress_confidence, 0.9)
            
            return False, 0.0
        
        except Exception as e:
            logger.error(f"Error in emotion distress detection: {e}")
            return False, 0.0
    
    def get_helpline_info(self, country_code: str = "US") -> Dict[str, Any]:
        """
        Get appropriate helpline information.
        
        Args:
            country_code: ISO country code
        
        Returns:
            Helpline information dictionary
        """
        try:
            countries = self.helpline_data.get('countries', {})
            
            # Try to get country-specific helpline
            if country_code in countries:
                return countries[country_code]
            
            # Fallback to US helpline
            if 'US' in countries:
                return countries['US']
            
            # Final fallback to global info
            return self.helpline_data.get('global', {
                "name": "Crisis Support",
                "message": "Please reach out to local crisis support services in your area."
            })
        
        except Exception as e:
            logger.error(f"Error getting helpline info: {e}")
            return {"name": "Crisis Support", "message": "Please seek immediate help if you're in crisis."}

def detect_distress(text: str, emotion_window: Optional[List[FusedEmotion]] = None, 
                   session_id: str = "unknown") -> Tuple[bool, Optional[DistressAlert]]:
    """
    Main distress detection function.
    
    Args:
        text: Text message to analyze
        emotion_window: Recent emotion history
        session_id: Session identifier
    
    Returns:
        Tuple of (distress_detected, alert_details)
    """
    monitor = SafetyMonitor()
    
    # Check text-based distress
    text_distressed, text_confidence, matched_keywords = monitor.detect_text_distress(text)
    
    # Check emotion-based distress
    emotion_distressed, emotion_confidence = False, 0.0
    if emotion_window:
        emotion_distressed, emotion_confidence = monitor.detect_emotion_distress(emotion_window)
    
    # Determine overall distress
    is_distressed = text_distressed or emotion_distressed
    
    if is_distressed:
        # Create distress alert
        trigger_type = "text" if text_distressed else "emotion"
        if text_distressed and emotion_distressed:
            trigger_type = "combined"
        
        confidence = max(text_confidence, emotion_confidence)
        
        alert = DistressAlert(
            session_id=session_id,
            trigger_type=trigger_type,
            confidence=confidence,
            message_content=text if text_distressed else None,
            emotion_data={
                "emotion_confidence": emotion_confidence,
                "recent_valence": emotion_window[-1].fused_valence if emotion_window else None
            } if emotion_distressed else None
        )
        
        logger.warning(f"Distress detected in session {session_id}: {trigger_type} (confidence: {confidence:.2f})")
        return True, alert
    
    return False, None

def get_crisis_response(alert: DistressAlert, country_code: str = "US") -> Dict[str, Any]:
    """
    Generate appropriate crisis response including helpline information.
    
    Args:
        alert: Distress alert details
        country_code: User's country code
    
    Returns:
        Crisis response information
    """
    monitor = SafetyMonitor()
    helpline_info = monitor.get_helpline_info(country_code)
    
    response = {
        "distress": True,
        "message": "I notice you might be going through a difficult time. Your safety and wellbeing are important.",
        "helpline": helpline_info,
        "urgent_message": "If you're having thoughts of self-harm or suicide, please reach out for immediate help.",
        "alert_details": {
            "trigger_type": alert.trigger_type,
            "confidence": alert.confidence,
            "timestamp": alert.timestamp
        }
    }
    
    # Mark alert as having provided helpline info
    alert.helpline_provided = True
    
    return response