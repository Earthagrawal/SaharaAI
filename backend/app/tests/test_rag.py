"""Tests for RAG (Retrieval-Augmented Generation) system."""

import pytest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from app.core.rag import build_index, keyword_search, semantic_rerank, RAGSystem

class TestRAGSystem:
    """Test RAG functionality."""
    
    @pytest.fixture
    def temp_knowledge_base(self):
        """Create temporary knowledge base for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_dir = os.path.join(temp_dir, 'kb')
            index_dir = os.path.join(temp_dir, 'index')
            os.makedirs(kb_dir)
            
            # Create sample documents
            docs = {
                'emotions.txt': '''
                Emotions are complex psychological states that involve physiological arousal, 
                expressive behaviors, and conscious experience. Basic emotions include happiness, 
                sadness, anger, fear, surprise, and disgust. Understanding emotions helps in 
                building empathetic AI systems.
                ''',
                'ai_safety.md': '''
                # AI Safety Guidelines
                
                AI systems should prioritize user wellbeing and safety. This includes:
                - Detecting signs of distress or crisis
                - Providing appropriate resources and support
                - Maintaining user privacy and confidentiality
                - Avoiding harm through inappropriate responses
                ''',
                'conversation_tips.json': '''
                {
                  "active_listening": "Pay attention to both verbal and non-verbal cues",
                  "empathy": "Acknowledge and validate the user's feelings",
                  "support": "Offer helpful resources when appropriate",
                  "boundaries": "Know when to refer to professional help"
                }
                '''
            }
            
            for filename, content in docs.items():
                with open(os.path.join(kb_dir, filename), 'w') as f:
                    f.write(content.strip())
            
            yield kb_dir, index_dir
    
    def test_rag_system_initialization(self):
        """Test RAG system initialization."""
        rag = RAGSystem()
        assert hasattr(rag, 'index_dir')
        assert hasattr(rag, 'embeddings_dir')
        assert hasattr(rag, 'model')
    
    def test_text_chunking(self):
        """Test text chunking functionality."""
        rag = RAGSystem()
        
        text = "This is a test document. " * 100  # Long text
        chunks = rag._chunk_text(text, chunk_size=10, overlap=2)
        
        assert len(chunks) > 1
        assert all('text' in chunk for chunk in chunks)
        assert all('chunk_id' in chunk for chunk in chunks)
        assert all(len(chunk['text'].split()) <= 10 for chunk in chunks)
    
    def test_build_index(self, temp_knowledge_base):
        """Test building search index from knowledge base."""
        kb_dir, index_dir = temp_knowledge_base
        
        build_index(kb_dir, index_dir)
        
        # Check that index was created
        assert os.path.exists(index_dir)
        
        # Should have created index files
        index_files = os.listdir(index_dir)
        assert len(index_files) > 0
    
    def test_keyword_search(self, temp_knowledge_base):
        """Test keyword search functionality."""
        kb_dir, index_dir = temp_knowledge_base
        
        # Build index first
        build_index(kb_dir, index_dir)
        
        # Test search
        with patch('app.core.rag.RAGSystem.index_dir', index_dir):
            results = keyword_search("emotions feelings", top_k=5)
            
            assert isinstance(results, list)
            
            if results:  # If search found results
                for result in results:
                    assert 'id' in result
                    assert 'content' in result
                    assert 'source' in result
                    assert 'score' in result
                    
                    # Should be relevant to emotions
                    content_lower = result['content'].lower()
                    assert any(word in content_lower for word in ['emotion', 'feeling', 'psychological'])
    
    def test_keyword_search_empty_index(self):
        """Test keyword search with non-existent index."""
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_index_dir = os.path.join(temp_dir, 'empty')
            
            with patch('app.core.rag.RAGSystem.index_dir', empty_index_dir):
                results = keyword_search("test query")
                assert results == []
    
    def test_semantic_rerank(self):
        """Test semantic reranking functionality."""
        # Create mock search results
        snippets = [
            {
                'id': 'doc1',
                'content': 'Machine learning algorithms for emotion detection',
                'source': 'ai_emotions.txt',
                'score': 0.8
            },
            {
                'id': 'doc2', 
                'content': 'Cooking recipes and food preparation techniques',
                'source': 'cooking.txt',
                'score': 0.6
            },
            {
                'id': 'doc3',
                'content': 'Emotional intelligence and empathy in human interactions',
                'source': 'psychology.txt',
                'score': 0.7
            }
        ]
        
        query = "emotion recognition artificial intelligence"
        
        # Test reranking
        reranked = semantic_rerank(query, snippets, top_k=2)
        
        assert isinstance(reranked, list)
        assert len(reranked) <= 2
        
        if reranked:
            # Should prioritize emotion-related content over cooking
            emotion_content = [r for r in reranked if 'emotion' in r['content'].lower()]
            cooking_content = [r for r in reranked if 'cooking' in r['content'].lower()]
            
            # Emotion content should be ranked higher than cooking content
            if emotion_content and cooking_content:
                emotion_semantic_score = emotion_content[0].get('semantic_score', 0)
                cooking_semantic_score = cooking_content[0].get('semantic_score', 0)
                assert emotion_semantic_score > cooking_semantic_score
    
    def test_semantic_rerank_no_model(self):
        """Test semantic reranking when no model is available."""
        snippets = [
            {'id': 'doc1', 'content': 'Test content 1', 'source': 'test1.txt'},
            {'id': 'doc2', 'content': 'Test content 2', 'source': 'test2.txt'}
        ]
        
        with patch('app.core.rag.RAGSystem.model', None):
            result = semantic_rerank("test query", snippets, top_k=1)
            
            # Should return original snippets when no model available
            assert len(result) == 1
            assert result[0]['id'] == snippets[0]['id']
    
    def test_rag_integration_workflow(self, temp_knowledge_base):
        """Test complete RAG workflow: index -> search -> rerank."""
        kb_dir, index_dir = temp_knowledge_base
        
        # 1. Build index
        build_index(kb_dir, index_dir)
        
        # 2. Perform keyword search
        with patch('app.core.rag.RAGSystem.index_dir', index_dir):
            search_results = keyword_search("AI safety guidelines", top_k=10)
        
        # 3. Rerank results
        if search_results:
            final_results = semantic_rerank("AI safety guidelines", search_results, top_k=3)
            
            assert isinstance(final_results, list)
            assert len(final_results) <= 3
            
            # Results should be relevant to AI safety
            for result in final_results:
                content_lower = result['content'].lower()
                assert any(word in content_lower for word in ['ai', 'safety', 'system', 'user'])
    
    @pytest.mark.integration 
    def test_sentence_transformer_integration(self):
        """Test sentence transformer model integration."""
        rag = RAGSystem()
        
        if rag.model:
            # Test encoding functionality
            text = "This is a test sentence for embedding generation"
            embedding = rag.model.encode(text)
            
            assert embedding is not None
            assert len(embedding.shape) == 1  # Should be 1D vector
            assert embedding.shape[0] > 0  # Should have some dimensions
        else:
            pytest.skip("Sentence transformer model not available")