"""LLM client for Google Gemini integration."""

import os
from typing import Iterator, Optional, List, Dict, Any
from google.cloud import aiplatform
from google.cloud.aiplatform import gapic
import google.auth
from google.auth.transport.requests import Request
import requests
import json

from ..config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

class GeminiClient:
    """Client for Google Gemini API integration."""
    
    def __init__(self):
        self.api_key = config.GEMINI_API_KEY
        self.model_name = "gemini-1.5-flash"
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
        if not self.api_key:
            logger.warning("DEMO MODE - Gemini API key not configured")
            self._demo_mode = True
        else:
            self._demo_mode = False
            logger.info("Gemini client initialized with API key")
    
    def _demo_response_generator(self, prompt: str) -> Iterator[str]:
        """Generate demo response for development/testing."""
        demo_responses = {
            "summarize": "DEMO SUMMARY: This is a demonstration summary of the provided content. The main points include key topics and important information extracted from the text.",
            "default": "DEMO RESPONSE: This is a demonstration response from Gemini 1.5 Flash. In production, this would be a real AI-generated response based on your input."
        }
        
        response = demo_responses.get("summarize" if "summarize" in prompt.lower() else "default")
        
        # Simulate streaming response
        words = response.split()
        for i, word in enumerate(words):
            if i == 0:
                yield word
            else:
                yield " " + word
    
    def _make_request(self, messages: List[Dict[str, str]], system: Optional[str] = None) -> Iterator[str]:
        """Make actual request to Gemini API."""
        try:
            # Prepare the request
            contents = []
            
            # Add system message if provided
            if system:
                contents.append({
                    "role": "user",
                    "parts": [{"text": f"System: {system}"}]
                })
            
            # Add conversation messages
            for msg in messages:
                role = "user" if msg.get("role") == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.get("content", "")}]
                })
            
            # API request payload
            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 2048,
                }
            }
            
            url = f"{self.base_url}/models/{self.model_name}:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
            
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            if "candidates" in result and len(result["candidates"]) > 0:
                content = result["candidates"][0]["content"]["parts"][0]["text"]
                # Simulate streaming by yielding chunks
                words = content.split()
                for i, word in enumerate(words):
                    if i == 0:
                        yield word
                    else:
                        yield " " + word
            else:
                yield "Error: No response generated"
                
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            yield f"Error: {str(e)}"

def chat(messages: List[Dict[str, str]], system: Optional[str] = None) -> Iterator[str]:
    """
    Generate chat response using Gemini.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        system: Optional system message
    
    Yields:
        Streaming response chunks
    """
    client = GeminiClient()
    
    if client._demo_mode:
        logger.warning("DEMO MODE - Using demo responses instead of Gemini API")
        prompt = " ".join([msg.get("content", "") for msg in messages])
        yield from client._demo_response_generator(prompt)
    else:
        yield from client._make_request(messages, system)

def summarize(texts: List[str]) -> str:
    """
    Summarize a list of texts using Gemini.
    
    Args:
        texts: List of text strings to summarize
    
    Returns:
        Summary string
    """
    client = GeminiClient()
    
    # Combine texts
    combined_text = "\n\n".join(texts)
    
    messages = [{
        "role": "user",
        "content": f"Please provide a concise summary of the following text:\n\n{combined_text}"
    }]
    
    # Collect streaming response
    response_chunks = []
    for chunk in chat(messages, "You are a helpful assistant that provides clear, concise summaries."):
        response_chunks.append(chunk)
    
    return "".join(response_chunks)