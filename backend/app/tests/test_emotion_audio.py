"""Tests for audio emotion detection."""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock

from app.core.emotion_audio import analyze_wav, analyze_stream, RivaEmotionClient

class TestAudioEmotion:
    """Test audio emotion detection functionality."""
    
    def test_demo_mode_initialization(self):
        """Test emotion client initialization in demo mode."""
        client = RivaEmotionClient()
        assert client.demo_mode is True
    
    def test_analyze_wav_demo_mode(self):
        """Test WAV file emotion analysis in demo mode."""
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(b'dummy audio data')
            temp_path = temp_file.name
        
        try:
            result = analyze_wav(temp_path)
            
            # Validate demo emotion result structure
            assert isinstance(result, dict)
            assert 'valence' in result
            assert 'arousal' in result
            assert 'confidence' in result
            assert 'dominant_emotion' in result
            assert 'emotion_scores' in result
            assert 'modality' in result
            
            # Check value ranges
            assert -1.0 <= result['valence'] <= 1.0
            assert -1.0 <= result['arousal'] <= 1.0
            assert 0.0 <= result['confidence'] <= 1.0
            assert result['modality'] == 'audio'
            assert isinstance(result['emotion_scores'], dict)
        
        finally:
            os.unlink(temp_path)
    
    def test_analyze_wav_nonexistent(self):
        """Test emotion analysis with non-existent file."""
        result = analyze_wav("/nonexistent/file.wav")
        assert "error" in result
        assert "not found" in result["error"]
    
    def test_analyze_stream_demo_mode(self):
        """Test streaming emotion analysis in demo mode."""
        # Create dummy audio chunks
        chunks = [b'chunk1', b'chunk2', b'chunk3']
        
        results = list(analyze_stream(iter(chunks)))
        
        # Should return demo emotion results
        assert len(results) > 0
        
        for result in results:
            assert isinstance(result, dict)
            if 'error' not in result:
                assert 'valence' in result
                assert 'arousal' in result
                assert 'confidence' in result
                assert 'modality' in result
                assert result['modality'] == 'audio'
    
    def test_demo_emotion_consistency(self):
        """Test that demo emotions are consistent for same input."""
        # Create temporary file with same name
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(b'dummy audio data')
            temp_path = temp_file.name
        
        try:
            result1 = analyze_wav(temp_path)
            result2 = analyze_wav(temp_path)
            
            # Should return same demo result for same file
            assert result1['dominant_emotion'] == result2['dominant_emotion']
            assert result1['valence'] == result2['valence']
            assert result1['arousal'] == result2['arousal']
        
        finally:
            os.unlink(temp_path)
    
    def test_emotion_scores_validity(self):
        """Test that emotion scores are valid."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(b'test audio')
            temp_path = temp_file.name
        
        try:
            result = analyze_wav(temp_path)
            
            if 'emotion_scores' in result:
                emotion_scores = result['emotion_scores']
                
                # All scores should be between 0 and 1
                for emotion, score in emotion_scores.items():
                    assert 0.0 <= score <= 1.0
                    assert isinstance(emotion, str)
                
                # Dominant emotion should have highest score
                dominant = result['dominant_emotion']
                if dominant in emotion_scores:
                    dominant_score = emotion_scores[dominant]
                    other_scores = [score for emotion, score in emotion_scores.items() if emotion != dominant]
                    if other_scores:
                        assert dominant_score >= max(other_scores)
        
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.integration
    def test_riva_emotion_integration_placeholder(self):
        """Placeholder test for future Riva emotion integration."""
        client = RivaEmotionClient()
        assert client.demo_mode is True
        
        # When Riva is configured, this should test:
        # - Actual Riva emotion detection API calls
        # - Real emotion classification accuracy
        # - Streaming emotion analysis
        # - Error handling for API failures
        
        pytest.skip("Riva emotion detection not configured - using demo mode")