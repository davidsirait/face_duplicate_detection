"""Input validation for user uploads for the gradio app"""

import os
from pathlib import Path
from typing import Tuple


# Maximum file size in MB
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def validate_image_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate uploaded image file
    
    Args:
        file_path: Path to uploaded file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_path or file_path == "":
        return False, "No file provided"
    
    path = Path(file_path)
    
    # Check if file exists
    if not path.exists():
        return False, "File does not exist"
    
    # Check file extension
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Check file size
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
    
    return True, ""


def validate_person_name(name: str) -> Tuple[bool, str]:
    """
    Validate person name input
    
    Args:
        name: Person name string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not name or name.strip() == "":
        return False, "Person name is required"
    
    if len(name.strip()) < 2:
        return False, "Person name must be at least 2 characters"
    
    if len(name.strip()) > 100:
        return False, "Person name is too long (max 100 characters)"
    
    return True, ""