#!/usr/bin/env python3
"""
Comprehensive Backend Test Suite for Sahara AI Assistant
Tests all endpoints, error handling, and integration scenarios.
"""

import requests
import json
import sys
import time
from datetime import datetime
from typing import Dict, Any, List

class SaharaAPITester:
    def __init__(self, base_url="https://ai-assistant-102.preview.emergentagent.com"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.session_id = None
        
    def log(self, message: str, level: str = "INFO"):
        """Log test messages with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def run_test(self, name: str, method: str, endpoint: str, expected_status: int, 
                 data: Dict = None, headers: Dict = None) -> tuple:
        """Run a single API test and return success status and response"""
        url = f"{self.base_url}{endpoint}"
        if headers is None:
            headers = {'Content-Type': 'application/json'}
            
        self.tests_run += 1
        self.log(f"🔍 Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
                
            success = response.status_code == expected_status
            
            if success:
                self.tests_passed += 1
                self.log(f"✅ {name} - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    return True, response_data
                except:
                    return True, {"raw_response": response.text}
            else:
                self.log(f"❌ {name} - Expected {expected_status}, got {response.status_code}")
                self.log(f"   Response: {response.text[:200]}")
                return False, {}
                
        except requests.exceptions.Timeout:
            self.log(f"❌ {name} - Request timed out", "ERROR")
            return False, {}
        except Exception as e:
            self.log(f"❌ {name} - Error: {str(e)}", "ERROR")
            return False, {}
    
    def test_system_endpoints(self):
        """Test basic system endpoints"""
        self.log("=== Testing System Endpoints ===")
        
        # Test root endpoint
        success, response = self.run_test("Root Endpoint", "GET", "/", 200)
        if success:
            self.log(f"   System: {response.get('message', 'N/A')}")
            self.log(f"   Version: {response.get('version', 'N/A')}")
            self.log(f"   Demo Mode: {response.get('demo_mode', {})}")
        
        # Test health endpoint
        success, response = self.run_test("Health Check", "GET", "/health", 200)
        if success:
            self.log(f"   Health Status: {response.get('status', 'N/A')}")
    
    def test_chat_endpoint(self):
        """Test chat functionality with various scenarios"""
        self.log("=== Testing Chat Endpoint ===")
        
        # Basic chat test
        chat_data = {
            "message": "Hello, how are you?",
            "session_id": "test_session_001",
            "include_context": False
        }
        
        success, response = self.run_test("Basic Chat", "POST", "/api/chat", 200, chat_data)
        if success:
            self.session_id = response.get('session_id')
            self.log(f"   Response length: {len(response.get('response', ''))}")
            self.log(f"   Session ID: {self.session_id}")
            self.log(f"   Distress detected: {response.get('distress_detected', False)}")
        
        # Chat with context
        chat_with_context = {
            "message": "What did I just ask you?",
            "session_id": self.session_id or "test_session_001",
            "include_context": True
        }
        
        success, response = self.run_test("Chat with Context", "POST", "/api/chat", 200, chat_with_context)
        
        # Test distress detection
        distress_message = {
            "message": "I feel hopeless and don't want to live anymore",
            "session_id": "distress_test_session",
            "include_context": False
        }
        
        success, response = self.run_test("Distress Detection", "POST", "/api/chat", 200, distress_message)
        if success:
            self.log(f"   Distress detected: {response.get('distress_detected', False)}")
            if response.get('helpline_info'):
                self.log(f"   Helpline provided: Yes")
        
        # Test invalid chat request
        invalid_chat = {"invalid_field": "test"}
        self.run_test("Invalid Chat Request", "POST", "/api/chat", 422, invalid_chat)
    
    def test_transcription_endpoint(self):
        """Test speech-to-text functionality"""
        self.log("=== Testing Transcription Endpoint ===")
        
        # Test with demo audio path (should work in demo mode)
        transcribe_data = {
            "audio_file_path": "/tmp/demo_audio.wav"
        }
        
        success, response = self.run_test("Audio Transcription", "POST", "/api/transcribe", 200, transcribe_data)
        if success:
            self.log(f"   Transcription: {response.get('transcription', 'N/A')}")
        
        # Test with invalid path
        invalid_transcribe = {
            "audio_file_path": "/nonexistent/path.wav"
        }
        
        # This might return 500 or 200 depending on demo mode implementation
        self.run_test("Invalid Audio Path", "POST", "/api/transcribe", 500, invalid_transcribe)
    
    def test_tts_endpoint(self):
        """Test text-to-speech functionality"""
        self.log("=== Testing TTS Endpoint ===")
        
        # Test basic TTS
        tts_data = {
            "text": "Hello, this is a test of the text to speech system.",
            "voice": "default"
        }
        
        success, response = self.run_test("Text-to-Speech", "POST", "/api/synthesize", 200, tts_data)
        if success:
            self.log(f"   Audio path: {response.get('audio_path', 'N/A')}")
            self.log(f"   Audio size: {response.get('size', 'N/A')} bytes")
        
        # Test empty text
        empty_tts = {
            "text": "",
            "voice": "default"
        }
        
        # This should either work (return empty audio) or fail gracefully
        self.run_test("Empty Text TTS", "POST", "/api/synthesize", 200, empty_tts)
    
    def test_emotion_analysis(self):
        """Test emotion analysis functionality"""
        self.log("=== Testing Emotion Analysis ===")
        
        # Test audio emotion analysis
        audio_emotion_data = {
            "audio_file_path": "/tmp/demo_audio.wav"
        }
        
        success, response = self.run_test("Audio Emotion Analysis", "POST", "/api/emotion/analyze", 200, audio_emotion_data)
        if success and response.get('fused_emotion'):
            emotion = response['fused_emotion']
            self.log(f"   Detected emotion: {emotion.get('emotion', 'N/A')}")
            self.log(f"   Confidence: {emotion.get('confidence', 'N/A')}")
        
        # Test video emotion analysis with dummy frame data
        video_emotion_data = {
            "video_frame": [[[255, 0, 0] for _ in range(100)] for _ in range(100)]  # 100x100 red frame
        }
        
        success, response = self.run_test("Video Emotion Analysis", "POST", "/api/emotion/analyze", 200, video_emotion_data)
        
        # Test multimodal emotion analysis
        multimodal_data = {
            "audio_file_path": "/tmp/demo_audio.wav",
            "video_frame": [[[0, 255, 0] for _ in range(50)] for _ in range(50)]  # 50x50 green frame
        }
        
        success, response = self.run_test("Multimodal Emotion Analysis", "POST", "/api/emotion/analyze", 200, multimodal_data)
        
        # Test with no data
        empty_emotion = {}
        success, response = self.run_test("Empty Emotion Request", "POST", "/api/emotion/analyze", 200, empty_emotion)
    
    def test_rag_system(self):
        """Test RAG (Retrieval-Augmented Generation) functionality"""
        self.log("=== Testing RAG System ===")
        
        # Test knowledge search
        rag_data = {
            "query": "What is Sahara AI Assistant?",
            "top_k": 3
        }
        
        success, response = self.run_test("RAG Knowledge Search", "POST", "/api/rag/search", 200, rag_data)
        if success:
            results = response.get('results', [])
            self.log(f"   Results found: {len(results)}")
            self.log(f"   Total found: {response.get('total_found', 0)}")
            if results:
                self.log(f"   First result source: {results[0].get('source', 'N/A')}")
        
        # Test with different query
        tech_query = {
            "query": "emotion detection multimodal",
            "top_k": 5
        }
        
        success, response = self.run_test("Technical Query Search", "POST", "/api/rag/search", 200, tech_query)
        
        # Test empty query
        empty_query = {
            "query": "",
            "top_k": 3
        }
        
        success, response = self.run_test("Empty Query Search", "POST", "/api/rag/search", 200, empty_query)
    
    def test_memory_system(self):
        """Test memory management functionality"""
        self.log("=== Testing Memory System ===")
        
        session_id = self.session_id or "test_memory_session"
        
        # Test getting recent memory
        memory_get_data = {
            "session_id": session_id,
            "action": "get_recent",
            "k": 5
        }
        
        success, response = self.run_test("Get Recent Memory", "POST", "/api/memory", 200, memory_get_data)
        if success:
            turns = response.get('turns', [])
            self.log(f"   Recent turns: {len(turns)}")
        
        # Test persisting session
        memory_persist_data = {
            "session_id": session_id,
            "action": "persist"
        }
        
        success, response = self.run_test("Persist Session Memory", "POST", "/api/memory", 200, memory_persist_data)
        if success:
            self.log(f"   Persist message: {response.get('message', 'N/A')}")
        
        # Test invalid action
        invalid_memory = {
            "session_id": session_id,
            "action": "invalid_action"
        }
        
        self.run_test("Invalid Memory Action", "POST", "/api/memory", 400, invalid_memory)
    
    def test_session_summary(self):
        """Test session summary functionality"""
        self.log("=== Testing Session Summary ===")
        
        session_id = self.session_id or "test_session_001"
        
        # Test getting session summary
        success, response = self.run_test("Session Summary", "GET", f"/api/sessions/{session_id}/summary", 200)
        if success:
            self.log(f"   Turn count: {response.get('turn_count', 0)}")
            self.log(f"   Summary length: {len(response.get('summary', ''))}")
            self.log(f"   Latest timestamp: {response.get('latest_timestamp', 'N/A')}")
        
        # Test non-existent session
        self.run_test("Non-existent Session Summary", "GET", "/api/sessions/nonexistent_session/summary", 404)
    
    def test_error_scenarios(self):
        """Test various error scenarios and edge cases"""
        self.log("=== Testing Error Scenarios ===")
        
        # Test malformed JSON
        try:
            response = requests.post(f"{self.base_url}/api/chat", 
                                   data="invalid json", 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=10)
            if response.status_code == 422:
                self.tests_passed += 1
                self.log("✅ Malformed JSON handling - Status: 422")
            else:
                self.log(f"❌ Malformed JSON handling - Expected 422, got {response.status_code}")
            self.tests_run += 1
        except Exception as e:
            self.log(f"❌ Malformed JSON test error: {e}")
            self.tests_run += 1
        
        # Test non-existent endpoint
        self.run_test("Non-existent Endpoint", "GET", "/api/nonexistent", 404)
        
        # Test method not allowed
        self.run_test("Method Not Allowed", "POST", "/health", 405)
    
    def run_all_tests(self):
        """Run the complete test suite"""
        start_time = time.time()
        self.log("🌵 Starting Sahara AI Assistant Backend Test Suite")
        self.log(f"Testing against: {self.base_url}")
        
        try:
            self.test_system_endpoints()
            self.test_chat_endpoint()
            self.test_transcription_endpoint()
            self.test_tts_endpoint()
            self.test_emotion_analysis()
            self.test_rag_system()
            self.test_memory_system()
            self.test_session_summary()
            self.test_error_scenarios()
            
        except KeyboardInterrupt:
            self.log("Test suite interrupted by user", "WARNING")
        except Exception as e:
            self.log(f"Unexpected error in test suite: {e}", "ERROR")
        
        # Print final results
        end_time = time.time()
        duration = end_time - start_time
        
        self.log("=" * 50)
        self.log("🏁 TEST SUITE COMPLETED")
        self.log(f"📊 Tests passed: {self.tests_passed}/{self.tests_run}")
        self.log(f"⏱️  Duration: {duration:.2f} seconds")
        
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        self.log(f"📈 Success rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            self.log("🎉 Backend system is functioning well!")
            return 0
        elif success_rate >= 60:
            self.log("⚠️  Backend system has some issues but is mostly functional")
            return 1
        else:
            self.log("🚨 Backend system has significant issues")
            return 2

def main():
    """Main test execution"""
    tester = SaharaAPITester()
    return tester.run_all_tests()

if __name__ == "__main__":
    sys.exit(main())