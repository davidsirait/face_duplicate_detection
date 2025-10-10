"""
Real-time face duplicate detection and inference
"""
import sys
from pathlib import Path

# Add src to path so imports work when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict

from core.config import Config
from core.pipeline import FaceProcessingPipeline

logger = logging.getLogger(__name__)


class FaceDuplicateDetector:
    """Real-time duplicate detection with improved error handling"""
    
    def __init__(self, config_path: str = "./config.yaml"):
        """Initialize duplicate detector"""
        try:
            self.config = Config(config_path)
            self.pipeline = FaceProcessingPipeline(self.config)
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            raise RuntimeError(f"Detector initialization failed: {e}")
    
    def check_duplicate(self, image_path: Path, n_results: int = 6) -> Tuple[bool, Optional[Dict], Optional[Dict]]:
        """
        Check if image contains a duplicate face
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not image_path or not Path(image_path).exists():
                return False, {"error": "Invalid or missing image path"}, None
            
            # Process image through pipeline
            success, embedding, metadata = self.pipeline.process_single_image(image_path)

            if not success:
                error_msg = metadata.get('error', 'Unknown processing error')
                logger.warning(f"Image processing failed: {error_msg}")
                return False, metadata, None
            
            if embedding is None:
                return False, {"error": "Failed to extract embedding"}, None
            
            # Search in database for duplicates
            try:
                is_dup, match_info = self.pipeline.db.check_duplicate(
                    embedding, threshold=self.config.get("duplicate_detection.threshold")
                )
            except Exception as e:
                logger.error(f"Database query failed: {e}")
                return False, {"error": f"Database error: {str(e)}"}, None

            # Search for top matches available in the database
            try:
                top_matches = self.pipeline.db.search(embedding, n_results)
            except Exception as e:
                logger.error(f"Top matches search failed: {e}")
                top_matches = {'ids': [[]], 'distances': [[]], 'metadatas': [[]]}
            
            processing_time = time.time() - start_time
            
            if match_info:
                match_info['processing_time'] = processing_time
                match_info['query_image'] = str(image_path)
            
            return is_dup, match_info, top_matches
            
        except Exception as e:
            logger.error(f"Unexpected error in check_duplicate: {e}")
            return False, {"error": f"Unexpected error: {str(e)}"}, None
    
    def add_to_database(self, image_path: Path, person_id: Optional[str] = None):
        """
        Add new face to database
        """
        try:
            success, embedding, metadata = self.pipeline.process_single_image(image_path)
            
            if success and embedding is not None:
                try:
                    doc_id = self.pipeline.db.add_embedding(
                        embedding, str(image_path), person_id, metadata
                    )
                    return True, doc_id
                except Exception as e:
                    logger.error(f"Database insertion failed: {e}")
                    return False, {"error": f"Database error: {str(e)}"}
            
            return False, metadata
            
        except Exception as e:
            logger.error(f"Error adding to database: {e}")
            return False, {"error": f"Unexpected error: {str(e)}"}
