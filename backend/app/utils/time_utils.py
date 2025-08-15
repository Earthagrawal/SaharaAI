"""Time utilities for Sahara application."""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pytz

def get_current_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()

def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime object."""
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        return None

def is_recent(timestamp: str, minutes: int = 30) -> bool:
    """Check if timestamp is within the last N minutes."""
    try:
        ts = parse_timestamp(timestamp)
        if ts:
            return datetime.now() - ts.replace(tzinfo=None) <= timedelta(minutes=minutes)
    except Exception:
        pass
    return False

def format_duration(start_time: str, end_time: Optional[str] = None) -> str:
    """Format duration between two timestamps."""
    try:
        start = parse_timestamp(start_time)
        end_dt = parse_timestamp(end_time) if end_time else datetime.now()
        
        if start and end_dt:
            duration = end_dt.replace(tzinfo=None) - start.replace(tzinfo=None)
            total_seconds = int(duration.total_seconds())
            
            if total_seconds < 60:
                return f"{total_seconds}s"
            elif total_seconds < 3600:
                return f"{total_seconds // 60}m {total_seconds % 60}s"
            else:
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                return f"{hours}h {minutes}m"
    except Exception:
        pass
    return "Unknown duration"