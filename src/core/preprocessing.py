"""
Image preprocessing module for face recognition
"""
from PIL import Image, ImageOps
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handles image preprocessing for face recognition"""
    
    def __init__(self, max_size=2000):
        """
        Initialize preprocessor
        
        Args:
            max_size: Maximum image dimension (width or height)
        """
        self.max_size = max_size
    
    def preprocess(self, image_path):
        """
        Preprocess image before face detection
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL.Image: Preprocessed image or None if failed
        """
        try:
            # Load image
            img = Image.open(image_path)
            
            # Fix EXIF orientation (important for phone photos)
            img = ImageOps.exif_transpose(img)
            
            # Ensure RGB format
            if img.mode == 'RGBA':
                # Create white background for transparency
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize large images for performance
            if max(img.size) > self.max_size:
                img.thumbnail((self.max_size, self.max_size), Image.LANCZOS)
            
            return img
            
        except Exception as e:
            logger.error(f"Error preprocessing {image_path}: {e}")
            return None