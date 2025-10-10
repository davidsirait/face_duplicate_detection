"""
End-to-end pipeline for processing face images
"""
from pathlib import Path
from tqdm import tqdm
import time
import torch  
import gc  
from typing import Optional, Dict
import logging

from core.config import Config
from core.preprocessing import ImagePreprocessor
from core.detector import FaceDetector
from core.embedding import FaceNetEmbedding
from db.vector_db import FaceVectorDB

logger = logging.getLogger(__name__)


class FaceProcessingPipeline:
    """End-to-end pipeline with memory management"""
    
    def __init__(self, config: Config):
        """Initialize pipeline with all components"""
        self.config = config
        
        # Initialize components
        self.preprocessor = ImagePreprocessor()
        
        # Face detector
        if config.get("face_detection.detector_type") == "mtcnn":
            self.detector = FaceDetector(device=config.get("device.type"))
        else:
            self.detector = FaceDetector(
                detector_type="yunet",
                yunet_model_path=config.get("paths.yunet_model_path"),
                device=config.get("device.type")
            )
        
        # Embedding extractor
        self.embedding_extractor = FaceNetEmbedding(
            model_name=config.get("embedding.embedding_model"),
            device=config.get("device.type")
        )

        # Vector database
        self.db = FaceVectorDB(
            persist_directory=str(config.get("paths.db_dir")),
            collection_name=config.get("vector_database.collection_name")
        )

        print(f"current records count : {self.db.get_count()}")
    
    def process_single_image(self, image_path: Path):
        """Process single image through entire pipeline"""
        start_time = time.time()
        
        # 1. Preprocess image
        img = self.preprocessor.preprocess(image_path)
        if img is None:
            return False, None, {"error": "Preprocessing failed"}
        
        # 2. Detect and align face
        face_tensor = self.detector.detect_and_align(img)
        if face_tensor is None:
            return False, None, {"error": "No face detected"}
        
        # 3. Extract embedding
        embedding = self.embedding_extractor.extract(face_tensor)
        if embedding is None:
            return False, None, {"error": "Embedding extraction failed"}
        
        processing_time = time.time() - start_time
        
        metadata = {
            "processing_time": processing_time,
            "detector": self.config.get("face_detection.detector_type")
        }
        
        return True, embedding, metadata
    
    @staticmethod
    def parse_person_id_from_filename(filename: str) -> str:
        """Extract person ID from filename"""
        parts = Path(filename).stem.split('_')
        if parts:
            return parts[0]
        return "unknown"
    
    def build_database(self, image_folder: str = "./data/images"):
        """
        Build database from folder of images
        """
        # Get all image files
        image_folder_path = Path(image_folder)
        image_paths = list(image_folder_path.glob("*.jpg")) + \
                     list(image_folder_path.glob("*.png")) + \
                     list(image_folder_path.glob("*.jpeg"))
        
        print(f"\nProcessing {len(image_paths)} images...")
        print(f"Detector: {self.config.get('face_detection.detector_type')}")
        print(f"Device: {self.config.get('device.type')}\n")
        
        successful = 0
        failed = 0
        no_face = 0
        
        embeddings_batch = []
        paths_batch = []
        ids_batch = []
        metadatas_batch = []
        
        batch_size = self.config.get("batch_processing.batch_size")
        
        for img_path in tqdm(image_paths, desc="Processing images"):
            person_id = FaceProcessingPipeline.parse_person_id_from_filename(str(img_path))
            success, embedding, metadata = self.process_single_image(img_path)
            
            if success:
                embeddings_batch.append(embedding)
                paths_batch.append(str(img_path))
                ids_batch.append(person_id)
                metadatas_batch.append(metadata)
                successful += 1
                
                # Batch insert when batch is full
                if len(embeddings_batch) >= batch_size:
                    self.db.add_embeddings_batch(
                        embeddings_batch, paths_batch, ids_batch, metadatas_batch
                    )
                    
                    # cleanup the lists after successful insert
                    embeddings_batch = []
                    paths_batch = []
                    ids_batch = []
                    metadatas_batch = []
                    
                    # Clear CUDA cache if using GPU
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Force garbage collection
                    gc.collect()
            else:
                if "No face detected" in metadata.get("error", ""):
                    no_face += 1
                else:
                    failed += 1
        
        # Insert remaining embeddings
        if embeddings_batch:
            try:
                self.db.add_embeddings_batch(
                    embeddings_batch, paths_batch, ids_batch, metadatas_batch
                )
            except Exception as e:
                logger.error(f"Error during adding embedding to db : {e}")
            
            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Database Build Complete")
        print(f"{'='*60}")
        print(f"Successful: {successful}")
        print(f"No face detected: {no_face}")
        print(f"Failed: {failed}")
        print(f"Total in database: {self.db.get_count()}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    config = Config()
    pipeline_ = FaceProcessingPipeline(config)
    pipeline_.build_database()