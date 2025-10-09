"""
Real-time face duplicate detection and inference
"""
import time
from pathlib import Path
from typing import Tuple, Optional, Dict

from config import Config
from pipeline import FaceProcessingPipeline


class FaceDuplicateDetector:
    """Real-time duplicate detection """
    
    def __init__(self, config_path: str  = "./config.yaml"):
        """
        Initialize duplicate detector
        
        Args:
            config: Configuration object
        """
        self.config = Config(config_path)
        self.pipeline = FaceProcessingPipeline(self.config)
    
    def check_duplicate(self, image_path: Path, n_results:int = 6) -> Tuple[bool, Optional[Dict], Optional[Dict]]:
        """
        Check if image contains a duplicate face
        
        Args:
            image_path: Path to image file
            n_result: number of similiar results to return

        Returns:
            Tuple[bool, Optional[Dict], Optional[Dict]]: (is_duplicate, match_info, top_matches)
                - is_duplicate: True if duplicate found
                - match_info: Dictionary with match details or error info
                - top_matches : Metadata information of most similiar results 
        """
        start_time = time.time()
        
        # Process image through pipeline
        success, embedding, metadata = self.pipeline.process_single_image(image_path)

        if not success:
            return False, metadata
        
        # Search in database for duplicates
        is_dup, match_info = self.pipeline.db.check_duplicate(
            embedding, threshold=self.config.get("duplicate_detection.threshold")
        )

        # search for top matches available in the database
        top_matches = self.pipeline.db.search(embedding, n_results)
        
        processing_time = time.time() - start_time
        
        if match_info:
            match_info['processing_time'] = processing_time
            match_info['query_image'] = str(image_path)
        
        return is_dup, match_info, top_matches
    
    def add_to_database(self, image_path: Path, person_id: Optional[str] = None):
        """
        Add new face to database
        
        Args:
            image_path: Path to image file
            person_id: Optional person identifier
            
        Returns:
            Tuple[bool, str/dict]: (success, document_id or error_info)
        """
        success, embedding, metadata = self.pipeline.process_single_image(image_path)
        
        if success:
            doc_id = self.pipeline.db.add_embedding(
                embedding, str(image_path), person_id, metadata
            )
            return True, doc_id
        
        return False, metadata
    
   