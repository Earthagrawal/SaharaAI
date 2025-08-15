#!/usr/bin/env python3
"""Demo script to showcase Sahara system capabilities."""

import os
import sys
import json
import tempfile
from datetime import datetime

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core.stt import transcribe_file
from app.core.emotion_audio import analyze_wav
from app.core.emotion_video import infer_frame
from app.core.llm_client import chat, summarize
from app.core.rag import keyword_search, semantic_rerank
from app.core.memory_manager import append_turn, persist_session
from app.core.safety import detect_distress, get_crisis_response
from app.services.file_store import FileStore
from app.utils.logger import get_logger
import numpy as np

logger = get_logger(__name__)

def demo_stt():
    """Demo speech-to-text functionality."""
    print("\n=== Speech-to-Text Demo ===")
    
    # Create dummy audio file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file.write(b'dummy audio data for demo')
        temp_path = temp_file.name
    
    try:
        result = transcribe_file(temp_path)
        print(f"STT Result: {result}")
        return result
    finally:
        os.unlink(temp_path)

def demo_emotion_audio():
    """Demo audio emotion detection."""
    print("\n=== Audio Emotion Detection Demo ===")
    
    # Create dummy audio file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file.write(b'dummy audio data for emotion analysis')
        temp_path = temp_file.name
    
    try:
        result = analyze_wav(temp_path)
        print(f"Audio Emotion: {json.dumps(result, indent=2)}")
        return result
    finally:
        os.unlink(temp_path)

def demo_emotion_video():
    """Demo video emotion detection."""
    print("\n=== Video Emotion Detection Demo ===")
    
    # Create dummy video frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    result = infer_frame(frame)
    print(f"Video Emotion: {json.dumps(result, indent=2)}")
    return result

def demo_llm():
    """Demo LLM chat functionality."""
    print("\n=== LLM Chat Demo ===")
    
    messages = [
        {"role": "user", "content": "Hello, how are you today?"}
    ]
    
    print("Chat Response: ", end="")
    response_chunks = []
    for chunk in chat(messages, "You are a helpful and empathetic AI assistant."):
        print(chunk, end="", flush=True)
        response_chunks.append(chunk)
    
    full_response = "".join(response_chunks)
    print(f"\n\nFull response: {full_response}")
    return full_response

def demo_summarization():
    """Demo text summarization."""
    print("\n=== Summarization Demo ===")
    
    texts = [
        "The user expressed feeling overwhelmed with work responsibilities and mentioned having trouble sleeping.",
        "I provided some suggestions for stress management including deep breathing exercises and time management tips.",
        "The user seemed receptive to the advice and asked for more information about meditation techniques."
    ]
    
    summary = summarize(texts)
    print(f"Summary: {summary}")
    return summary 

def demo_rag():
    """Demo RAG (search and retrieval) functionality."""
    print("\n=== RAG System Demo ===")
    
    query = "emotion detection AI systems"
    
    # Perform keyword search
    search_results = keyword_search(query, top_k=5)
    print(f"Search results for '{query}': {len(search_results)} found")
    
    if search_results:
        # Rerank results
        reranked = semantic_rerank(query, search_results, top_k=3)
        print(f"Top reranked results:")
        for i, result in enumerate(reranked[:2]):
            print(f"  {i+1}. {result.get('content', '')[:100]}...")
    
    return search_results

def demo_memory():
    """Demo memory management."""
    print("\n=== Memory Management Demo ===")
    
    session_id = f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Add some conversation turns
    turns = [
        {
            "user_message": "I'm feeling a bit stressed about work lately",
            "assistant_response": "I understand that work stress can be challenging. Would you like to talk about what's been causing you the most stress?",
            "emotion_summary": "mild stress"
        },
        {
            "user_message": "My manager has been giving me too many tasks",
            "assistant_response": "That sounds overwhelming. Have you considered discussing your workload with your manager?",
            "emotion_summary": "frustrated"
        },
        {
            "user_message": "Thanks for listening, I feel better now",
            "assistant_response": "I'm glad I could help. Remember that it's important to take care of your mental health.",
            "emotion_summary": "relieved"
        }
    ]
    
    for turn in turns:
        append_turn(session_id, turn)
        print(f"Added turn: {turn['user_message'][:50]}...")
    
    # Persist session
    persist_session(session_id)
    print(f"Persisted session {session_id} to long-term memory")
    
    return session_id

def demo_safety():
    """Demo safety detection system."""
    print("\n=== Safety Detection Demo ===")
    
    # Test with non-distressing message
    safe_message = "I'm having a good day and feeling positive about things"
    distressed, alert = detect_distress(safe_message)
    print(f"Safe message analysis - Distressed: {distressed}")
    
    # Test with concerning message (demo purposes only)
    concerning_message = "I'm feeling really hopeless and don't know if I can go on"
    distressed, alert = detect_distress(concerning_message, session_id="demo_safety")
    print(f"Concerning message analysis - Distressed: {distressed}")
    
    if distressed and alert:
        crisis_response = get_crisis_response(alert)
        print(f"Crisis response activated: {crisis_response.get('message', '')[:100]}...")
        print(f"Helpline provided: {crisis_response.get('helpline', {}).get('name', 'N/A')}")
    
    return distressed, alert

def demo_file_storage():
    """Demo file storage system."""
    print("\n=== File Storage Demo ===")
    
    file_store = FileStore()
    
    # Save demo data
    demo_data = {
        "session_id": "demo_file_session",
        "timestamp": datetime.now().isoformat(),
        "data": "This is demo data for file storage testing"
    }
    
    file_path = file_store.save_json(demo_data, "demo_data.json", "temp")
    print(f"Saved demo data to: {file_path}")
    
    # Load the data back
    loaded_data = file_store.load_json("demo_data.json", "temp")
    if loaded_data:
        print(f"Loaded data: {loaded_data['data']}")
    
    return file_path

def main():
    """Run complete demo workflow."""
    print("🌵 Sahara AI Assistant - Demo Mode")
    print("=" * 50)
    
    try:
        # Run all demo components
        demo_stt()
        demo_emotion_audio()
        demo_emotion_video()
        demo_llm()
        demo_summarization()
        demo_rag()
        session_id = demo_memory()
        demo_safety()
        demo_file_storage()
        
        print("\n" + "=" * 50)
        print("✅ Demo completed successfully!")
        print(f"Demo session ID: {session_id}")
        print("\nAll Sahara components are working in demo mode.")
        print("Configure Riva credentials in .env to enable full functionality.")
        
        return 0
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())