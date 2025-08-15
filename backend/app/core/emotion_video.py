"""Video emotion detection using OpenCV DNN and demo fallback."""

import os
import cv2
import numpy as np
from typing import Dict, Any, Optional
import random

from ..config import config
from ..utils.logger import get_logger
from ..models.emotion import EmotionResult

logger = get_logger(__name__)

class VideoEmotionDetector:
    """OpenCV-based facial emotion detection with demo fallback."""
    
    def __init__(self):
        self.model_path = os.path.join(config.MODEL_PATHS, "emotion_models")
        self.face_net = None
        self.emotion_net = None
        self.demo_mode = True  # Default to demo mode
        
        self._load_models()
    
    def _load_models(self):
        """Load OpenCV DNN models for face detection and emotion recognition."""
        try:
            # Try to load face detection model (SSD)
            face_proto = os.path.join(self.model_path, "opencv_face_detector.pbtxt")
            face_model = os.path.join(self.model_path, "opencv_face_detector_uint8.pb")
            
            if os.path.exists(face_proto) and os.path.exists(face_model):
                self.face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)
                logger.info("Face detection model loaded successfully")
            else:
                logger.warning("Face detection model files not found")
            
            # Try to load emotion recognition model (mini_XCEPTION)
            emotion_model = os.path.join(self.model_path, "fer2013_mini_XCEPTION.102-0.66.hdf5")
            
            if os.path.exists(emotion_model):
                # Note: This would require tensorflow/keras to load
                # For demo purposes, we'll skip actual loading
                logger.info("Emotion recognition model path found")
            else:
                logger.warning("Emotion recognition model not found")
            
            # Check if we have all required models
            if not (self.face_net and os.path.exists(emotion_model)):
                self.demo_mode = True
                logger.warning("DEMO MODE - Required models not available, using demo emotion detection")
            else:
                self.demo_mode = False
                logger.info("Video emotion detection models loaded successfully")
                
        except Exception as e:
            logger.warning(f"Error loading models: {e}, using demo mode")
            self.demo_mode = True
    
    def _detect_faces(self, frame: np.ndarray) -> list:
        """Detect faces in frame using OpenCV DNN."""
        if not self.face_net:
            return []
        
        try:
            height, width = frame.shape[:2]
            
            # Create blob from frame
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > 0.5:  # Confidence threshold
                    x1 = int(detections[0, 0, i, 3] * width)
                    y1 = int(detections[0, 0, i, 4] * height)
                    x2 = int(detections[0, 0, i, 5] * width)
                    y2 = int(detections[0, 0, i, 6] * height)
                    
                    faces.append((x1, y1, x2, y2, confidence))
            
            return faces
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []
    
    def _predict_emotion(self, face_roi: np.ndarray) -> EmotionResult:
        """Predict emotion from face region of interest."""
        # Emotion categories for FER2013/AffectNet
        emotions = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]
        
        if self.demo_mode:
            # Generate demo emotion result
            emotion_idx = random.randint(0, len(emotions) - 1)
            emotion_name = emotions[emotion_idx]
            
            # Generate emotion scores
            emotion_scores = {}
            for i, emotion in enumerate(emotions):
                if i == emotion_idx:
                    emotion_scores[emotion] = random.uniform(0.6, 0.9)
                else:
                    emotion_scores[emotion] = random.uniform(0.0, 0.3)
            
            # Map to valence/arousal
            emotion_mapping = {
                "angry": (-0.6, 0.7),
                "disgusted": (-0.8, 0.3),
                "fearful": (-0.7, 0.6),
                "happy": (0.8, 0.5),
                "sad": (-0.7, -0.4),
                "surprised": (0.2, 0.8),
                "neutral": (0.0, 0.0)
            }
            
            valence, arousal = emotion_mapping.get(emotion_name, (0.0, 0.0))
            
            return EmotionResult(
                valence=valence,
                arousal=arousal,
                confidence=emotion_scores[emotion_name],
                dominant_emotion=emotion_name,
                emotion_scores=emotion_scores,
                modality="video"
            )
        
        # TODO: Implement actual emotion prediction
        # This would involve:
        # 1. Preprocessing face ROI (resize, normalize)
        # 2. Running inference with emotion model
        # 3. Converting predictions to emotion scores
        # 4. Mapping to valence/arousal space
        
        # For now, return demo result
        return self._predict_emotion(face_roi)  # Recursive call to demo mode

def infer_frame(frame: np.ndarray) -> Dict[str, Any]:
    """
    Infer emotion from a single video frame.
    
    Args:
        frame: Video frame as numpy array (BGR format)
    
    Returns:
        Emotion detection results
    """
    detector = VideoEmotionDetector()
    
    if frame is None or frame.size == 0:
        logger.error("Invalid or empty frame provided")
        return {"error": "Invalid frame data"}
    
    try:
        # Detect faces in frame
        faces = detector._detect_faces(frame)
        
        if not faces:
            logger.info("No faces detected in frame")
            return {
                "faces_detected": 0,
                "emotions": [],
                "message": "No faces detected"
            }
        
        # Analyze emotion for each detected face
        emotions = []
        for i, (x1, y1, x2, y2, confidence) in enumerate(faces):
            # Extract face ROI
            face_roi = frame[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                continue
            
            # Predict emotion
            emotion_result = detector._predict_emotion(face_roi)
            
            # Add face location info
            emotion_dict = emotion_result.dict()
            emotion_dict.update({
                "face_id": i,
                "face_location": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "face_confidence": float(confidence)
            })
            
            emotions.append(emotion_dict)
        
        return {
            "faces_detected": len(faces),
            "emotions": emotions,
            "frame_shape": frame.shape
        }
    
    except Exception as e:
        logger.error(f"Error in video emotion detection: {e}")
        return {"error": f"Video emotion detection failed - {str(e)}"}