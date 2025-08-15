"""Configuration management for Sahara backend."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for Sahara application."""
    
    # Google Gemini Configuration
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    
    # NVIDIA Riva Configuration
    RIVA_URL: Optional[str] = os.getenv("RIVA_URL")
    RIVA_API_KEY: Optional[str] = os.getenv("RIVA_API_KEY")
    
    # Data and Model Paths
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    MODEL_PATHS: str = os.getenv("MODEL_PATHS", "./models")
    
    # MongoDB Configuration (existing)
    MONGO_URL: str = os.getenv("MONGO_URL", "mongodb://localhost:27017/sahara")
    
    # Derived paths
    MEMORY_SUMMARY_PATH: str = os.path.join(DATA_DIR, "memory_summary.txt")
    USER_PROFILE_PATH: str = os.path.join(DATA_DIR, "user_profile.json")
    CONVERSATIONS_DIR: str = os.path.join(DATA_DIR, "conversations")
    EMBEDDINGS_DIR: str = os.path.join(DATA_DIR, "embeddings")
    TEMP_AUDIO_DIR: str = os.path.join(DATA_DIR, "temp_audio")
    KNOWLEDGE_BASE_DIR: str = os.path.join(DATA_DIR, "knowledge_base")
    HELPLINES_PATH: str = os.path.join(DATA_DIR, "helplines.json")
    
    # Ensure directories exist
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.CONVERSATIONS_DIR,
            cls.EMBEDDINGS_DIR,
            cls.TEMP_AUDIO_DIR,
            cls.KNOWLEDGE_BASE_DIR
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# Global config instance
config = Config()
config.ensure_directories()