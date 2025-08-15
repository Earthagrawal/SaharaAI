"""File storage service for managing data persistence."""

import os
import json
import shutil
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid

from ..config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

class FileStore:
    """Local filesystem storage service."""
    
    def __init__(self):
        self.data_dir = config.DATA_DIR
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [
            config.CONVERSATIONS_DIR,
            config.EMBEDDINGS_DIR,
            config.TEMP_AUDIO_DIR,
            config.KNOWLEDGE_BASE_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def save_json(self, data: Dict[str, Any], filename: str, subdirectory: str = "") -> str:
        """
        Save data as JSON file.
        
        Args:
            data: Dictionary to save
            filename: Name of the file
            subdirectory: Optional subdirectory within data_dir
        
        Returns:
            Full path to saved file
        """
        try:
            if subdirectory:
                save_dir = os.path.join(self.data_dir, subdirectory)
                os.makedirs(save_dir, exist_ok=True)
            else:
                save_dir = self.data_dir
            
            file_path = os.path.join(save_dir, filename)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Saved JSON data to {file_path}")
            return file_path
        
        except Exception as e:
            logger.error(f"Error saving JSON file: {e}")
            raise
    
    def load_json(self, filename: str, subdirectory: str = "") -> Optional[Dict[str, Any]]:
        """
        Load data from JSON file.
        
        Args:
            filename: Name of the file
            subdirectory: Optional subdirectory within data_dir
        
        Returns:
            Loaded data or None if file doesn't exist
        """
        try:
            if subdirectory:
                file_path = os.path.join(self.data_dir, subdirectory, filename)
            else:
                file_path = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded JSON data from {file_path}")
            return data
        
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            return None
    
    def save_audio(self, audio_data: bytes, filename: Optional[str] = None) -> str:
        """
        Save audio data to temp directory.
        
        Args:
            audio_data: Audio bytes
            filename: Optional filename, generates UUID if not provided
        
        Returns:
            Full path to saved audio file
        """
        try:
            if not filename:
                filename = f"audio_{uuid.uuid4().hex}.wav"
            
            file_path = os.path.join(config.TEMP_AUDIO_DIR, filename)
            
            with open(file_path, 'wb') as f:
                f.write(audio_data)
            
            logger.info(f"Saved audio data to {file_path}")
            return file_path
        
        except Exception as e:
            logger.error(f"Error saving audio file: {e}")
            raise
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """
        Clean up old temporary files.
        
        Args:
            max_age_hours: Maximum age of files to keep
        """
        try:
            current_time = datetime.now()
            cleaned_count = 0
            
            for root, dirs, files in os.walk(config.TEMP_AUDIO_DIR):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        age_hours = (current_time - file_time).total_seconds() / 3600
                        
                        if age_hours > max_age_hours:
                            os.remove(file_path)
                            cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Error checking file {file_path}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} temporary files")
        
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_file_info(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a file.
        
        Args:
            filepath: Path to the file
        
        Returns:
            File information dictionary or None
        """
        try:
            if not os.path.exists(filepath):
                return None
            
            stat = os.stat(filepath)
            return {
                "path": filepath,
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "is_file": os.path.isfile(filepath),
                "is_directory": os.path.isdir(filepath)
            }
        
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return None