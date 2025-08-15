"""Tests for video emotion detection."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from app.core.emotion_video import infer_frame, VideoEmotionDetector

class TestVideoEmotion:
    """Test video emotion detection functionality."""
    
    def test_demo_mode_initialization(self):
        """Test video emotion detector initialization in demo mode."""
        detector = VideoEmotionDetector()
        assert detector.demo_mode is True
    
    def test_infer_frame_valid_input(self):
        """Test frame emotion inference with valid input."""
        # Create dummy frame (BGR format)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = infer_frame(frame)
        
        # Validate result structure
        assert isinstance(result, dict)
        
        if 'error' not in result:
            assert 'faces_detected' in result
            assert 'emotions' in result
            assert isinstance(result['faces_detected'], int)
            assert isinstance(result['emotions'], list)
            
            # If faces were detected, check emotion structure
            for emotion in result['emotions']:
                assert 'valence' in emotion
                assert 'arousal' in emotion
                assert 'confidence' in emotion
                assert 'dominant_emotion' in emotion
                assert 'modality' in emotion
                assert emotion['modality'] == 'video'
                
                # Check value ranges
                assert -1.0 <= emotion['valence'] <= 1.0
                assert -1.0 <= emotion['arousal'] <= 1.0
                assert 0.0 <= emotion['confidence'] <= 1.0
    
    def test_infer_frame_invalid_input(self):
        """Test frame emotion inference with invalid input."""
        # Test with None
        result = infer_frame(None)
        assert 'error' in result
        
        # Test with empty array
        result = infer_frame(np.array([]))
        assert 'error' in result
    
    def test_infer_frame_grayscale(self):
        """Test frame emotion inference with grayscale input."""
        # Create grayscale frame
        frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        result = infer_frame(frame)
        
        # Should handle grayscale input
        assert isinstance(result, dict)
        # May not detect faces in random noise, but shouldn't error
        if 'error' not in result:
            assert 'faces_detected' in result
    
    def test_emotion_mapping_consistency(self):
        """Test that emotion mappings are consistent."""
        detector = VideoEmotionDetector()
        
        # Create dummy face ROI
        face_roi = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
        
        # Test multiple predictions for consistency
        result1 = detector._predict_emotion(face_roi)
        result2 = detector._predict_emotion(face_roi)
        
        # In demo mode, should return consistent results based on input
        assert result1.dominant_emotion == result2.dominant_emotion
        assert result1.valence == result2.valence
        assert result1.arousal == result2.arousal
    
    def test_face_detection_demo(self):
        """Test face detection in demo mode."""
        detector = VideoEmotionDetector()
        
        # Create frame with some structure (not just noise)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some rectangular structures that might look like faces
        frame[100:200, 200:300] = [128, 128, 128]  # Gray rectangle
        
        faces = detector._detect_faces(frame)
        
        # In demo mode without actual models, this may return empty list
        assert isinstance(faces, list)
        
        # Each detected face should have proper format
        for face in faces:
            assert len(face) == 5  # x1, y1, x2, y2, confidence
            x1, y1, x2, y2, conf = face
            assert x1 < x2
            assert y1 < y2
            assert 0.0 <= conf <= 1.0
    
    def test_emotion_categories(self):
        """Test that all expected emotion categories are present."""
        detector = VideoEmotionDetector()
        face_roi = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
        
        result = detector._predict_emotion(face_roi)
        
        expected_emotions = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]
        
        # Check that dominant emotion is from expected set
        assert result.dominant_emotion in expected_emotions
        
        # Check that emotion scores contain expected emotions
        for emotion in expected_emotions:
            assert emotion in result.emotion_scores
            assert 0.0 <= result.emotion_scores[emotion] <= 1.0
    
    @pytest.mark.integration
    def test_opencv_integration_placeholder(self):
        """Placeholder test for future OpenCV model integration."""
        detector = VideoEmotionDetector()
        assert detector.demo_mode is True
        
        # When OpenCV models are available, this should test:
        # - Actual face detection using DNN
        # - Real emotion classification with mini_XCEPTION
        # - Model loading and inference performance
        # - Accuracy on known test cases
        
        pytest.skip("OpenCV emotion models not available - using demo mode")