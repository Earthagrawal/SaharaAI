"""Tests for memory management system."""

import pytest
import os
import tempfile
import json
from unittest.mock import patch, MagicMock

from app.core.memory_manager import append_turn, get_recent, persist_session, MemoryManager
from app.models.memory import ConversationTurn

class TestMemoryManager:
    """Test memory management functionality."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('app.config.config.DATA_DIR', temp_dir):
                with patch('app.config.config.CONVERSATIONS_DIR', os.path.join(temp_dir, 'conversations')):
                    with patch('app.config.config.MEMORY_SUMMARY_PATH', os.path.join(temp_dir, 'memory_summary.txt')):
                        yield temp_dir
    
    def test_memory_manager_initialization(self, temp_data_dir):
        """Test memory manager initialization."""
        manager = MemoryManager()
        assert manager.max_short_term == 10  # default value
        assert hasattr(manager, 'short_term_memory')
        assert hasattr(manager, 'conversation_dir')
    
    def test_append_turn(self, temp_data_dir):
        """Test appending a conversation turn."""
        session_id = "test_session"
        turn_data = {
            "user_message": "Hello, how are you?",
            "assistant_response": "I'm doing well, thank you!",
            "emotion_summary": "positive"
        }
        
        append_turn(session_id, turn_data)
        
        # Check that turn was added to memory
        recent_turns = get_recent(session_id, 1)
        assert len(recent_turns) == 1
        assert recent_turns[0]['user_message'] == turn_data['user_message']
        assert recent_turns[0]['assistant_response'] == turn_data['assistant_response']
        assert 'timestamp' in recent_turns[0]  # Should be auto-added
    
    def test_get_recent_turns(self, temp_data_dir):
        """Test retrieving recent conversation turns."""
        session_id = "test_session"
        
        # Add multiple turns
        for i in range(5):
            turn_data = {
                "user_message": f"Message {i}",
                "assistant_response": f"Response {i}"
            }
            append_turn(session_id, turn_data)
        
        # Get recent turns
        recent_3 = get_recent(session_id, 3)
        assert len(recent_3) == 3
        assert recent_3[0]['user_message'] == "Message 2"  # Should be oldest of the 3
        assert recent_3[2]['user_message'] == "Message 4"  # Should be newest
    
    def test_persist_session(self, temp_data_dir):
        """Test persisting session to long-term memory."""
        session_id = "test_session"
        
        # Add some conversation turns
        turns = [
            {"user_message": "I'm feeling sad today", "assistant_response": "I'm sorry to hear that. Would you like to talk about it?"},
            {"user_message": "Yes, I had a difficult day at work", "assistant_response": "Work stress can be challenging. What happened?"},
            {"user_message": "My project got rejected", "assistant_response": "That must be disappointing. How are you coping with it?"}
        ]
        
        for turn in turns:
            append_turn(session_id, turn)
        
        # Persist the session
        persist_session(session_id)
        
        # Check that summary was written to file
        memory_summary_path = os.path.join(temp_data_dir, 'memory_summary.txt')
        assert os.path.exists(memory_summary_path)
        
        with open(memory_summary_path, 'r') as f:
            content = f.read()
            assert len(content) > 0
            # Should contain date, tags, and summary
            lines = content.strip().split('\n')
            assert len(lines) >= 1
            
            # Parse the summary line
            parts = lines[0].split('|')
            assert len(parts) == 3  # date|tags|summary
    
    def test_conversation_persistence(self, temp_data_dir):
        """Test that conversations are persisted to disk."""
        session_id = "test_session"
        turn_data = {
            "user_message": "Test message",
            "assistant_response": "Test response"
        }
        
        append_turn(session_id, turn_data)
        
        # Check that conversation file was created
        conv_dir = os.path.join(temp_data_dir, 'conversations')
        conv_file = os.path.join(conv_dir, f"{session_id}.json")
        
        assert os.path.exists(conv_file)
        
        # Load and verify content
        with open(conv_file, 'r') as f:
            history = json.load(f)
            assert len(history) == 1
            assert history[0]['user_message'] == turn_data['user_message']
    
    def test_memory_summarization(self, temp_data_dir):
        """Test memory summarization functionality."""
        manager = MemoryManager()
        
        # Create test turns with emotional content
        turns = [
            {
                "user_message": "I'm feeling overwhelmed with work",
                "assistant_response": "That sounds stressful. Can you tell me more?",
                "emotion_summary": "stressed"
            },
            {
                "user_message": "I have too many deadlines",
                "assistant_response": "Managing multiple deadlines can be challenging",
                "emotion_summary": "anxious"
            }
        ]
        
        # Test both LLM and fallback summarization
        summary = manager._extract_summary(turns)
        assert isinstance(summary, str)
        assert len(summary) > 0
        
        # Should contain relevant information
        summary_lower = summary.lower()
        assert any(word in summary_lower for word in ['work', 'stress', 'deadline', 'overwhelm'])
    
    def test_short_term_memory_limit(self, temp_data_dir):
        """Test that short-term memory respects size limits."""
        session_id = "test_session"
        manager = MemoryManager(max_short_term=3)
        
        # Add more turns than the limit
        for i in range(5):
            turn_data = {"user_message": f"Message {i}"}
            manager.short_term_memory[session_id].append(turn_data)
        
        # Should only keep the last 3
        assert len(manager.short_term_memory[session_id]) == 3
        messages = [turn['user_message'] for turn in manager.short_term_memory[session_id]]
        assert messages == ["Message 2", "Message 3", "Message 4"]
    
    def test_tag_extraction(self, temp_data_dir):
        """Test tag extraction from conversation history."""
        from app.core.memory_manager import _extract_tags
        
        history = [
            {
                "user_message": "I need help with my anxiety",
                "emotion_summary": "negative distress"
            },
            {
                "user_message": "How can I manage stress better?",
                "emotion_summary": "neutral"
            },
            {
                "user_message": "Thank you, that was helpful",
                "emotion_summary": "positive"
            }
        ]
        
        tags = _extract_tags(history)
        
        assert isinstance(tags, list)
        assert len(tags) <= 3  # Should limit to 3 tags
        # Should identify relevant tags
        assert any(tag in ['emotional', 'support', 'inquiry', 'positive'] for tag in tags)