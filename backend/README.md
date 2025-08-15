# Sahara AI Assistant

An advanced AI assistant with multimodal emotion detection, empathetic conversation capabilities, and built-in safety monitoring.

## Features

### Core Capabilities
- **Multimodal Emotion Detection**: Audio (voice) and video (facial) emotion recognition
- **Contextual Memory**: Short-term and long-term conversation memory management
- **RAG System**: Knowledge base integration with semantic search and reranking
- **Safety Monitoring**: Crisis detection and helpline resource provision
- **Empathetic Responses**: LLM-powered conversations with emotional awareness

### Technical Components
- **Speech Processing**: STT and TTS using NVIDIA Riva (with demo fallbacks)
- **Computer Vision**: OpenCV-based facial emotion recognition
- **LLM Integration**: Google Gemini 1.5 Flash for natural language processing
- **Search & Retrieval**: Whoosh + Sentence Transformers + FAISS vector search
- **Safety System**: Text and emotion-based distress detection

## Quick Start

### Prerequisites
- Python 3.11+
- Required dependencies (see requirements.txt)

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Initialize data directories:**
   ```bash
   python -c "from app.config import config; config.ensure_directories()"
   ```

4. **Build knowledge base index:**
   ```bash
   python scripts/build_whoosh_index.py
   ```

### Running the System

#### Demo Mode (No API Keys Required)
```bash
# Run demo script
python scripts/run_demo.py

# Start FastAPI server
python server.py
```

#### Production Mode
1. Configure API keys in `.env`:
   - `GEMINI_API_KEY`: Google Gemini API key
   - `RIVA_URL`: NVIDIA Riva server URL
   - `RIVA_API_KEY`: NVIDIA Riva API key

2. Download required models:
   ```bash
   bash scripts/download_models.sh
   ```

3. Start the server:
   ```bash
   python server.py
   ```

## Configuration

### Environment Variables (.env)
```env
# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# NVIDIA Riva Configuration
RIVA_URL=your_riva_server_url
RIVA_API_KEY=your_riva_api_key

# Data Paths
DATA_DIR=./data
MODEL_PATHS=./models
```

### Directory Structure
```
sahara/
├── backend/
│   ├── app/
│   │   ├── core/           # Core AI components
│   │   ├── models/         # Data models
│   │   ├── services/       # Service layer
│   │   ├── utils/          # Utilities
│   │   └── tests/          # Unit tests
│   ├── data/               # Data storage
│   │   ├── conversations/  # Chat history
│   │   ├── embeddings/     # Vector embeddings
│   │   ├── knowledge_base/ # Knowledge documents
│   │   └── temp_audio/     # Temporary audio files
│   ├── scripts/            # Utility scripts
│   └── server.py           # FastAPI application
```

## API Endpoints

### Chat
```http
POST /api/chat
Content-Type: application/json

{
  "message": "Hello, how are you?",
  "session_id": "optional_session_id",
  "include_context": true,
  "emotion_context": {...}
}
```

### Speech Processing
```http
POST /api/transcribe
POST /api/synthesize
```

### Emotion Analysis
```http
POST /api/emotion/analyze
```

### Knowledge Search
```http
POST /api/rag/search
```

### Memory Management
```http
POST /api/memory
GET /api/sessions/{session_id}/summary
```

## Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest app/tests/test_stt.py
pytest app/tests/test_emotion_audio.py
pytest app/tests/test_memory.py
```

## Safety & Privacy

### Crisis Detection
The system monitors for:
- Distress keywords in text messages
- Sustained negative emotions across modalities
- Crisis indicators in conversation patterns

When distress is detected:
- Helpline resources are provided immediately
- No external notifications are sent (privacy-preserving)
- Crisis support information is displayed to the user

### Data Privacy
- **Local Storage**: All data stored locally in plain text files
- **No Encryption**: Data is not encrypted (host-level security recommended)
- **User Control**: Users have full control over their data
- **No External Sharing**: Data never leaves the local system

### Host-Level Security Recommendations
- Use encrypted filesystems (e.g., LUKS, BitLocker)
- Implement proper file permissions (chmod 600 for sensitive files)  
- Regular backups with encryption
- Network security (firewall, VPN access)
- Physical security for the hosting machine

## Demo Mode vs Production

### Demo Mode (Default)
- Works without external API keys
- Uses deterministic demo responses
- All components functional with simulated outputs
- Perfect for development and testing

### Production Mode
- Requires Google Gemini API key
- Optional NVIDIA Riva integration
- Real AI model inference
- Full multimodal capabilities

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Missing Directories**: Run config.ensure_directories()
3. **Model Loading**: Check MODEL_PATHS configuration
4. **API Failures**: Verify API keys and network connectivity

### Logs
Check application logs for detailed error information:
```bash
tail -f logs/sahara.log  # If logging to file
```

### Debug Mode
Set logging level to DEBUG in utils/logger.py for verbose output.

## Contributing

1. Follow existing code structure and patterns
2. Add tests for new functionality
3. Update documentation for API changes
4. Ensure demo mode compatibility

## License

This project is designed for educational and research purposes. Please ensure compliance with API provider terms of service.