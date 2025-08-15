"""Tests for Speech-to-Text functionality."""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock

from app.core.stt import transcribe_file, transcribe_stream, RivaSTTClient

class TestSTT:
    """Test Speech-to-Text functionality."""
    
    def test_demo_mode_initialization(self):
        """Test STT client initialization in demo mode."""
        client = RivaSTTClient()
        assert client.demo_mode is True  # Should be True without Riva credentials
    
    def test_transcribe_file_demo_mode(self):
        """Test file transcription in demo mode."""
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(b'dummy audio data')
            temp_path = temp_file.name
        
        try:
            result = transcribe_file(temp_path)
            
            # Should return a demo transcription
            assert isinstance(result, str)
            assert len(result) > 0
            assert "demonstration" in result.lower() or "demo" in result.lower()
        
        finally:
            os.unlink(temp_path)
    
    def test_transcribe_file_nonexistent(self):
        """Test transcription with non-existent file."""
        result = transcribe_file("/nonexistent/file.wav")
        assert "Error:" in result
        assert "not found" in result
    
    def test_transcribe_stream_demo_mode(self):
        """Test streaming transcription in demo mode."""
        # Create dummy audio chunks
        chunks = [b'chunk1', b'chunk2', b'chunk3']
        
        results = list(transcribe_stream(iter(chunks)))
        
        # Should return demo streaming results
        assert len(results) > 0
        assert all(isinstance(result, str) for result in results)
    
    def test_demo_transcription_consistency(self):
        """Test that demo transcriptions are consistent for same input."""
        # Create temporary files with same name
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(b'dummy audio data')
            temp_path = temp_file.name
        
        try:
            result1 = transcribe_file(temp_path)
            result2 = transcribe_file(temp_path)
            
            # Should return same demo result for same file
            assert result1 == result2
        
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.integration
    def test_riva_integration_placeholder(self):
        """Placeholder test for future Riva integration."""
        # This test would verify actual Riva integration when credentials are available
        # For now, it just ensures the demo mode works correctly
        
        client = RivaSTTClient()
        assert client.demo_mode is True
        
        # When Riva is configured, this should test:
        # - Actual gRPC connection to Riva
        # - Real transcription accuracy
        # - Error handling for connection issues
        # - Streaming functionality
        
        pytest.skip("Riva integration not configured - using demo mode")