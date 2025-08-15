"""Main FastAPI server for Sahara AI Assistant."""

import os
import uuid
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# Import Sahara components
from app.config import config
from app.utils.logger import get_logger
from app.models.chat import ChatRequest, ChatResponse, ChatMessage
from app.models.emotion import EmotionResult, FusedEmotion, fuse_emotions
from app.core.llm_client import chat as llm_chat
from app.core.stt import transcribe_file
from app.core.tts import synthesize
from app.core.emotion_audio import analyze_wav
from app.core.emotion_video import infer_frame
from app.core.memory_manager import append_turn, get_recent, persist_session
from app.core.rag import keyword_search, semantic_rerank
from app.core.safety import detect_distress, get_crisis_response
from app.services.file_store import FileStore
from app.services.summarizer import summarize

logger = get_logger(__name__)

# Request/Response models
class TranscribeRequest(BaseModel):
    audio_file_path: str

class TTSRequest(BaseModel):
    text: str
    voice: str = "default"

class EmotionAnalysisRequest(BaseModel):
    audio_file_path: Optional[str] = None
    video_frame: Optional[List[List[List[int]]]] = None  # 3D array as nested lists

class RAGRequest(BaseModel):
    query: str
    top_k: int = 3

class MemoryRequest(BaseModel):
    session_id: str
    action: str  # "get_recent", "persist"
    k: Optional[int] = 5

# Global storage for active sessions
active_sessions: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("🌵 Starting Sahara AI Assistant")
    logger.info(f"Data directory: {config.DATA_DIR}")
    logger.info(f"Gemini API configured: {'Yes' if config.GEMINI_API_KEY else 'No'}")
    logger.info(f"Riva URL configured: {'Yes' if config.RIVA_URL else 'No (Demo Mode)'}")
    
    # Ensure directories exist
    config.ensure_directories()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Sahara AI Assistant")

# Create FastAPI app
app = FastAPI(
    title="Sahara AI Assistant",
    description="Empathetic AI assistant with multimodal emotion detection and safety monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with system status."""
    return {
        "message": "🌵 Sahara AI Assistant",
        "status": "running",
        "version": "1.0.0",
        "demo_mode": {
            "gemini": not bool(config.GEMINI_API_KEY),
            "riva": not bool(config.RIVA_URL and config.RIVA_API_KEY)
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": config.DATA_DIR}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    """Main chat endpoint with emotion awareness and safety monitoring."""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Initialize session if new
        if session_id not in active_sessions:
            active_sessions[session_id] = {
                "emotion_history": [],
                "turn_count": 0
            }
        
        session = active_sessions[session_id]
        
        # Check for distress in the message
        emotion_window = session.get("emotion_history", [])
        distressed, alert = detect_distress(request.message, emotion_window, session_id)
        
        # Build context for LLM
        context_messages = []
        if request.include_context:
            # Get recent conversation history
            recent_turns = get_recent(session_id, 5)
            for turn in recent_turns:
                if turn.get('user_message'):
                    context_messages.append({"role": "user", "content": turn['user_message']})
                if turn.get('assistant_response'):
                    context_messages.append({"role": "assistant", "content": turn['assistant_response']})
        
        # Add current message
        context_messages.append({"role": "user", "content": request.message})
        
        # Enhance with RAG if available
        context_sources = []
        try:
            search_results = keyword_search(request.message, top_k=10)
            if search_results:
                reranked = semantic_rerank(request.message, search_results, top_k=3)
                if reranked:
                    knowledge_context = "\n".join([r['content'][:200] for r in reranked])
                    context_sources = [r['source'] for r in reranked]
                    
                    # Add knowledge context to system message
                    system_message = f"""You are Sahara, an empathetic AI assistant. Use this knowledge context when relevant:

{knowledge_context}

Always prioritize user wellbeing and provide supportive, helpful responses."""
                else:
                    system_message = "You are Sahara, an empathetic AI assistant focused on user wellbeing and support."
            else:
                system_message = "You are Sahara, an empathetic AI assistant focused on user wellbeing and support."
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")
            system_message = "You are Sahara, an empathetic AI assistant focused on user wellbeing and support."
        
        # Generate response
        response_chunks = []
        for chunk in llm_chat(context_messages, system_message):
            response_chunks.append(chunk)
        
        full_response = "".join(response_chunks)
        
        # Create turn record
        turn_data = {
            "user_message": request.message,
            "assistant_response": full_response,
            "emotion_context": request.emotion_context,
            "distress_detected": distressed
        }
        
        # Add turn to memory
        append_turn(session_id, turn_data)
        session["turn_count"] += 1
        
        # Prepare response
        response_data = {
            "response": full_response,
            "session_id": session_id,
            "distress_detected": distressed,
            "context_used": context_sources
        }
        
        # Add crisis response if needed
        if distressed and alert:
            crisis_info = get_crisis_response(alert)
            response_data["helpline_info"] = crisis_info.get("helpline")
            # Prepend crisis message to response
            crisis_message = crisis_info.get("message", "")
            response_data["response"] = f"{crisis_message}\n\n{full_response}"
        
        # Schedule background cleanup if session is long
        if session["turn_count"] > 50:
            background_tasks.add_task(persist_session, session_id)
        
        return ChatResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.post("/api/transcribe")
async def transcribe_endpoint(request: TranscribeRequest):
    """Speech-to-text transcription endpoint."""
    try:
        result = transcribe_file(request.audio_file_path)
        return {"transcription": result}
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/api/synthesize")
async def tts_endpoint(request: TTSRequest):
    """Text-to-speech synthesis endpoint."""
    try:
        audio_data = synthesize(request.text, request.voice)
        
        # Save audio temporarily and return path
        file_store = FileStore()
        audio_path = file_store.save_audio(audio_data)
        
        return {
            "audio_path": audio_path,
            "size": len(audio_data),
            "voice": request.voice
        }
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")

@app.post("/api/emotion/analyze")
async def emotion_analysis_endpoint(request: EmotionAnalysisRequest):
    """Multimodal emotion analysis endpoint."""
    try:
        voice_emotion = None
        face_emotion = None
        
        # Analyze audio emotion if provided
        if request.audio_file_path:
            audio_result = analyze_wav(request.audio_file_path)
            if 'error' not in audio_result:
                voice_emotion = EmotionResult(**audio_result)
        
        # Analyze video emotion if provided
        if request.video_frame:
            # Convert nested list to numpy array
            frame_array = np.array(request.video_frame, dtype=np.uint8)
            video_result = infer_frame(frame_array)
            
            if 'error' not in video_result and video_result.get('emotions'):
                # Use first detected face emotion
                first_emotion = video_result['emotions'][0]
                face_emotion = EmotionResult(**first_emotion)
        
        # Fuse emotions if multiple modalities available
        if voice_emotion or face_emotion:
            fused_emotion = fuse_emotions(voice_emotion, face_emotion)
            return {
                "fused_emotion": fused_emotion.dict(),
                "voice_emotion": voice_emotion.dict() if voice_emotion else None,
                "face_emotion": face_emotion.dict() if face_emotion else None
            }
        else:
            return {"message": "No valid emotion data detected"}
            
    except Exception as e:
        logger.error(f"Emotion analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Emotion analysis failed: {str(e)}")

@app.post("/api/rag/search")
async def rag_search_endpoint(request: RAGRequest):
    """Knowledge base search endpoint."""
    try:
        # Perform keyword search
        search_results = keyword_search(request.query, top_k=20)
        
        if not search_results:
            return {"results": [], "message": "No results found"}
        
        # Rerank with semantic similarity
        reranked_results = semantic_rerank(request.query, search_results, top_k=request.top_k)
        
        return {
            "results": reranked_results,
            "total_found": len(search_results),
            "returned": len(reranked_results)
        }
        
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/memory")
async def memory_endpoint(request: MemoryRequest):
    """Memory management endpoint."""
    try:
        if request.action == "get_recent":
            recent_turns = get_recent(request.session_id, request.k or 5)
            return {"turns": recent_turns}
        
        elif request.action == "persist":
            persist_session(request.session_id)
            return {"message": f"Session {request.session_id} persisted to long-term memory"}
        
        else:
            raise HTTPException(status_code=400, detail="Invalid action. Use 'get_recent' or 'persist'")
            
    except Exception as e:
        logger.error(f"Memory endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Memory operation failed: {str(e)}")

@app.get("/api/sessions/{session_id}/summary")
async def session_summary_endpoint(session_id: str):
    """Get session summary and statistics."""
    try:
        recent_turns = get_recent(session_id, 10)
        
        if not recent_turns:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Generate summary
        turn_texts = []
        for turn in recent_turns:
            if turn.get('user_message'):
                turn_texts.append(turn['user_message'])
            if turn.get('assistant_response'):
                turn_texts.append(turn['assistant_response'])
        
        summary_text = summarize(turn_texts) if turn_texts else "No conversation content"
        
        return {
            "session_id": session_id,
            "turn_count": len(recent_turns),
            "summary": summary_text,
            "latest_timestamp": recent_turns[-1].get('timestamp') if recent_turns else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session summary error: {e}")
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)