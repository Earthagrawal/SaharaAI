"""Audio processing utilities."""

import numpy as np
import scipy.signal
from typing import Tuple, Optional
import os

from ..utils.logger import get_logger

logger = get_logger(__name__)

def convert_to_16khz_mono(audio_data: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
    """
    Convert audio to 16kHz mono format.
    
    Args:
        audio_data: Audio samples
        sample_rate: Original sample rate
    
    Returns:
        Tuple of (converted_audio, new_sample_rate)
    """
    try:
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        target_rate = 16000
        
        # Resample if needed
        if sample_rate != target_rate:
            # Calculate number of samples for target rate
            num_samples = int(len(audio_data) * target_rate / sample_rate)
            audio_data = scipy.signal.resample(audio_data, num_samples)
        
        # Normalize to [-1, 1]
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                audio_data = audio_data.astype(np.float32)
        
        # Ensure values are in valid range
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        logger.info(f"Converted audio to 16kHz mono: {len(audio_data)} samples")
        return audio_data, target_rate
    
    except Exception as e:
        logger.error(f"Error converting audio: {e}")
        raise

def safe_truncate(audio_data: np.ndarray, max_duration_seconds: float = 300.0, sample_rate: int = 16000) -> np.ndarray:
    """
    Safely truncate audio to maximum duration.
    
    Args:
        audio_data: Audio samples
        max_duration_seconds: Maximum duration in seconds
        sample_rate: Sample rate
    
    Returns:
        Truncated audio data
    """
    max_samples = int(max_duration_seconds * sample_rate)
    
    if len(audio_data) > max_samples:
        logger.warning(f"Truncating audio from {len(audio_data)} to {max_samples} samples")
        audio_data = audio_data[:max_samples]
    
    return audio_data

def calculate_rms_energy(audio_data: np.ndarray, window_size: int = 1024) -> np.ndarray:
    """
    Calculate RMS energy of audio signal.
    
    Args:
        audio_data: Audio samples
        window_size: Window size for RMS calculation
    
    Returns:
        RMS energy values
    """
    try:
        # Pad audio if necessary
        if len(audio_data) < window_size:
            audio_data = np.pad(audio_data, (0, window_size - len(audio_data)), 'constant')
        
        # Calculate RMS for each window
        rms_values = []
        for i in range(0, len(audio_data) - window_size + 1, window_size // 2):
            window = audio_data[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms)
        
        return np.array(rms_values)
    
    except Exception as e:
        logger.error(f"Error calculating RMS energy: {e}")
        return np.array([0.0])

def detect_silence(audio_data: np.ndarray, threshold: float = 0.01, min_duration: float = 0.5, sample_rate: int = 16000) -> list:
    """
    Detect silent regions in audio.
    
    Args:
        audio_data: Audio samples
        threshold: RMS threshold for silence
        min_duration: Minimum duration for silence detection (seconds)
        sample_rate: Sample rate
    
    Returns:
        List of (start_time, end_time) tuples for silent regions
    """
    try:
        window_size = int(0.1 * sample_rate)  # 100ms windows
        rms_values = calculate_rms_energy(audio_data, window_size)
        
        # Find silent windows
        silent_windows = rms_values < threshold
        
        # Group consecutive silent windows
        silent_regions = []
        start_idx = None
        
        for i, is_silent in enumerate(silent_windows):
            if is_silent and start_idx is None:
                start_idx = i
            elif not is_silent and start_idx is not None:
                # End of silent region
                duration = (i - start_idx) * (window_size / 2) / sample_rate
                if duration >= min_duration:
                    start_time = start_idx * (window_size / 2) / sample_rate
                    end_time = i * (window_size / 2) / sample_rate
                    silent_regions.append((start_time, end_time))
                start_idx = None
        
        # Handle case where audio ends with silence
        if start_idx is not None:
            duration = (len(silent_windows) - start_idx) * (window_size / 2) / sample_rate
            if duration >= min_duration:
                start_time = start_idx * (window_size / 2) / sample_rate
                end_time = len(audio_data) / sample_rate
                silent_regions.append((start_time, end_time))
        
        logger.info(f"Detected {len(silent_regions)} silent regions")
        return silent_regions
    
    except Exception as e:
        logger.error(f"Error detecting silence: {e}")
        return []