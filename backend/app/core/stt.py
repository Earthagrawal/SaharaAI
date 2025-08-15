"""Speech-to-Text (STT) implementation with NVIDIA Riva integration and demo fallback."""

import os
from typing import Iterator, Optional
import grpc
from itertools import cycle

from ..config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

class RivaSTTClient:
    """NVIDIA Riva Speech-to-Text client with demo fallback."""
    
    def __init__(self):
        self.riva_url = config.RIVA_URL
        self.riva_api_key = config.RIVA_API_KEY
        self.demo_mode = not (self.riva_url and self.riva_api_key)
        
        if self.demo_mode:
            logger.warning("DEMO MODE - Riva STT not configured, using demo responses")
        else:
            logger.info("Riva STT client initialized")
    
    def _get_demo_transcription(self, file_path: str) -> str:
        """Generate demo transcription based on file name or content."""
        demo_transcriptions = [
            "This is a demonstration transcription of your audio file. In production, this would contain the actual speech-to-text conversion from NVIDIA Riva.",
            "Hello, this is a sample transcription. The actual system would process your audio using advanced speech recognition.",
            "Demo mode active. Your audio would be transcribed using NVIDIA Riva's state-of-the-art speech recognition technology.",
            "This represents what a real transcription would look like. The system is ready for Riva integration when you provide the API credentials."
        ]
        
        # Use file name hash to get consistent demo response
        file_hash = hash(os.path.basename(file_path)) % len(demo_transcriptions)
        return demo_transcriptions[file_hash]
    
    def _demo_streaming_transcription(self) -> Iterator[str]:
        """Generate demo streaming transcription."""
        words = [
            "This", "is", "a", "demo", "streaming", "transcription.",
            "In", "production,", "this", "would", "be", "real-time",
            "speech-to-text", "from", "NVIDIA", "Riva."
        ]
        
        for word in words:
            yield word + " "

def transcribe_file(path: str) -> str:
    """
    Transcribe audio file to text.
    
    Args:
        path: Path to audio file
    
    Returns:
        Transcribed text
    """
    client = RivaSTTClient()
    
    if not os.path.exists(path):
        logger.error(f"Audio file not found: {path}")
        return f"Error: Audio file not found at {path}"
    
    if client.demo_mode:
        logger.warning("DEMO MODE - Using demo transcription")
        return client._get_demo_transcription(path)
    
    # TODO: Implement actual Riva gRPC call
    try:
        # Placeholder for Riva implementation:
        # 1. Connect to Riva server using gRPC
        # 2. Configure recognition parameters
        # 3. Send audio file for transcription
        # 4. Return transcribed text
        
        logger.info(f"Transcribing file: {path}")
        # For now, return demo response
        return client._get_demo_transcription(path)
        
    except Exception as e:
        logger.error(f"Error in transcription: {e}")
        return f"Error: Transcription failed - {str(e)}"

def transcribe_stream(chunks: Iterator[bytes]) -> Iterator[str]:
    """
    Transcribe streaming audio to text.
    
    Args:
        chunks: Iterator of audio chunks (bytes)
    
    Yields:
        Partial transcription results
    """
    client = RivaSTTClient()
    
    if client.demo_mode:
        logger.warning("DEMO MODE - Using demo streaming transcription")
        yield from client._demo_streaming_transcription()
        return
    
    # TODO: Implement actual Riva streaming gRPC call
    try:
        # Placeholder for Riva streaming implementation:
        # 1. Establish streaming gRPC connection
        # 2. Send audio chunks as they arrive  
        # 3. Yield partial transcription results
        # 4. Handle end of stream
        
        logger.info("Starting streaming transcription")
        
        # For now, yield demo response
        yield from client._demo_streaming_transcription()
        
    except Exception as e:
        logger.error(f"Error in streaming transcription: {e}")
        yield f"Error: Streaming transcription failed - {str(e)}"