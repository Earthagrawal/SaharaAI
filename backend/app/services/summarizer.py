"""Summarization service using LLM with fallback methods."""

from typing import List, Dict, Any
from collections import Counter
import re

from ..core.llm_client import summarize as llm_summarize
from ..utils.logger import get_logger

logger = get_logger(__name__)

def summarize(turns: List[Dict[str, Any]]) -> str:
    """
    Summarize conversation turns using LLM or fallback method.
    
    Args:
        turns: List of conversation turn dictionaries
    
    Returns:
        Summary string
    """
    if not turns:
        return "No conversation content to summarize"
    
    try:
        # Extract text content from turns
        texts = []
        for turn in turns:
            if turn.get('user_message'):
                texts.append(f"User: {turn['user_message']}")
            if turn.get('assistant_response'):
                texts.append(f"Assistant: {turn['assistant_response']}")
            if turn.get('emotion_summary'):
                texts.append(f"Emotion: {turn['emotion_summary']}")
        
        if not texts:
            return "No textual content found in conversation"
        
        # Try LLM summarization first
        try:
            summary = llm_summarize(texts)
            if summary and not summary.startswith("Error:"):
                logger.info("Successfully generated LLM summary")
                return summary
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
        
        # Fallback to extractive summarization
        logger.info("Using fallback summarization method")
        return _extractive_summary(texts)
    
    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        return f"Summary generation failed: {str(e)}"

def _extractive_summary(texts: List[str]) -> str:
    """
    Fallback extractive summarization using frequency-based approach.
    
    Args:
        texts: List of text strings
    
    Returns:
        Extractive summary
    """
    try:
        # Combine all text
        combined_text = " ".join(texts)
        
        # Extract sentences
        sentences = re.split(r'[.!?]+', combined_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 3:
            return combined_text[:500] + ("..." if len(combined_text) > 500 else "")
        
        # Calculate word frequencies
        words = re.findall(r'\b\w+\b', combined_text.lower())
        word_freq = Counter(words)
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        word_freq = {word: freq for word, freq in word_freq.items() if word not in stop_words}
        
        # Score sentences based on word frequencies
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            words_in_sentence = re.findall(r'\b\w+\b', sentence.lower())
            score = sum(word_freq.get(word, 0) for word in words_in_sentence)
            sentence_scores[i] = score / max(len(words_in_sentence), 1)  # Normalize by length
        
        # Select top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_sentences = sorted(top_sentences, key=lambda x: x[0])  # Sort by original order
        
        summary_sentences = [sentences[i] for i, _ in top_sentences]
        return ". ".join(summary_sentences) + "."
    
    except Exception as e:
        logger.error(f"Error in extractive summarization: {e}")
        return "Unable to generate summary due to processing error"