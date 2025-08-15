"""Video processing utilities."""

import cv2
import numpy as np
from typing import Tuple, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)

def resize_frame(frame: np.ndarray, target_size: Tuple[int, int] = (640, 480)) -> np.ndarray:
    """
    Resize video frame to target dimensions.
    
    Args:
        frame: Input frame
        target_size: Target (width, height)
    
    Returns:
        Resized frame
    """
    try:
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame data")
        
        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        return resized
    
    except Exception as e:
        logger.error(f"Error resizing frame: {e}")
        raise

def preprocess_face_roi(face_roi: np.ndarray, target_size: Tuple[int, int] = (48, 48)) -> np.ndarray:
    """
    Preprocess face ROI for emotion recognition.
    
    Args:
        face_roi: Face region of interest
        target_size: Target size for emotion model
    
    Returns:
        Preprocessed face data
    """
    try:
        if face_roi is None or face_roi.size == 0:
            raise ValueError("Invalid face ROI")
        
        # Convert to grayscale if needed
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi
        
        # Resize to target size
        resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions for model input
        preprocessed = np.expand_dims(normalized, axis=0)  # Add batch dimension
        preprocessed = np.expand_dims(preprocessed, axis=-1)  # Add channel dimension
        
        return preprocessed
    
    except Exception as e:
        logger.error(f"Error preprocessing face ROI: {e}")
        raise

def extract_video_frames(video_path: str, max_frames: int = 100, skip_frames: int = 5) -> list:
    """
    Extract frames from video file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract
        skip_frames: Number of frames to skip between extractions
    
    Returns:
        List of extracted frames
    """
    frames = []
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        frame_count = 0
        extracted_count = 0
        
        while extracted_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Skip frames according to skip_frames parameter
            if frame_count % (skip_frames + 1) == 0:
                frames.append(frame)
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from video")
        return frames
    
    except Exception as e:
        logger.error(f"Error extracting video frames: {e}")
        return []

def calculate_frame_quality(frame: np.ndarray) -> float:
    """
    Calculate quality score for a video frame.
    
    Args:
        frame: Input frame
    
    Returns:
        Quality score (0.0 to 1.0)
    """
    try:
        if frame is None or frame.size == 0:
            return 0.0
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Calculate Laplacian variance (focus measure)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 range (empirically determined)
        quality_score = min(laplacian_var / 1000.0, 1.0)
        
        return quality_score
    
    except Exception as e:
        logger.error(f"Error calculating frame quality: {e}")
        return 0.0

def create_video_thumbnail(video_path: str, output_path: str, timestamp: float = 1.0) -> bool:
    """
    Create thumbnail from video at specified timestamp.
    
    Args:
        video_path: Path to input video
        output_path: Path for output thumbnail
        timestamp: Timestamp in seconds
    
    Returns:
        True if successful, False otherwise
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return False
        
        # Set position to timestamp
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Resize to reasonable thumbnail size
            thumbnail = resize_frame(frame, (320, 240))
            cv2.imwrite(output_path, thumbnail)
            logger.info(f"Created video thumbnail: {output_path}")
            return True
        
        return False
    
    except Exception as e:
        logger.error(f"Error creating video thumbnail: {e}")
        return False