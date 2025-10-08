"""
End-to-end pipeline for processing face images
"""
from pathlib import Path
from tqdm import tqdm
import time
from typing import Optional, Dict
import logging

from config import Config
from preprocessing import ImagePreprocessor
from detector import MTCNNDetector, YuNetDetector
from embedding import FaceNetEmbedding
from vector_db import FaceVectorDB

logger = logging.getLogger(__name__)


class FaceProcessingPipeline:
    """End-to-end pipeline for processing face images and building database"""
    
    def __init__(self, config: Config):
        """
        Initialize pipeline with all components
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize components
        self.preprocessor = ImagePreprocessor()
        
        # Face detector
        if config.DETECTOR_TYPE == "mtcnn":
            self.detector = MTCNNDetector(device=config.DEVICE)
        else:
            self.detector = YuNetDetector(
                model_path=config.YUNET_MODEL_PATH,
                device=config.DEVICE
            )
        
        # Embedding extractor
        self.embedding_extractor = FaceNetEmbedding(
            model_name=config.EMBEDDING_MODEL,
            device=config.DEVICE
        )
        
        # Vector database
        self.db = FaceVectorDB(
            persist_directory=str(config.DB_DIR),
            collection_name=config.COLLECTION_NAME
        )
    
    def process_single_image(self, image_path: Path, person_id: Optional[str] = None):
        """
        Process single image through entire pipeline
        
        Args:
            image_path: Path to image file
            person_id: Optional person identifier
            
        Returns:
            Tuple: (success: bool, embedding: np.ndarray, metadata: dict)
        """
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
            "detector": self.config.DETECTOR_TYPE
        }
        
        return True, embedding, metadata
    
    def build_database(self, image_folder: Path, person_id_mapping: Optional[Dict] = None):
        """
        Build database from folder of images
        
        Args:
            image_folder: Path to folder containing images
            person_id_mapping: Optional dict mapping image filename to person_id
        """
        # Get all image files
        image_paths = list(image_folder.glob("*.jpg")) + \
                     list(image_folder.glob("*.png")) + \
                     list(image_folder.glob("*.jpeg"))
        
        print(f"\nProcessing {len(image_paths)} images...")
        print(f"Detector: {self.config.DETECTOR_TYPE}")
        print(f"Device: {self.config.DEVICE}\n")
        
        successful = 0
        failed = 0
        no_face = 0
        
        embeddings_batch = []
        paths_batch = []
        ids_batch = []
        metadatas_batch = []
        
        for img_path in tqdm(image_paths, desc="Processing images"):
            person_id = None
            if person_id_mapping:
                person_id = person_id_mapping.get(img_path.name)
            
            success, embedding, metadata = self.process_single_image(img_path, person_id)
            
            if success:
                embeddings_batch.append(embedding)
                paths_batch.append(str(img_path))
                ids_batch.append(person_id)
                metadatas_batch.append(metadata)
                successful += 1
                
                # Batch insert when batch is full
                if len(embeddings_batch) >= self.config.BATCH_SIZE:
                    self.db.add_embeddings_batch(
                        embeddings_batch, paths_batch, ids_batch, metadatas_batch
                    )
                    embeddings_batch = []
                    paths_batch = []
                    ids_batch = []
                    metadatas_batch = []
            else:
                if "No face detected" in metadata.get("error", ""):
                    no_face += 1
                else:
                    failed += 1
        
        # Insert remaining embeddings
        if embeddings_batch:
            self.db.add_embeddings_batch(
                embeddings_batch, paths_batch, ids_batch, metadatas_batch
            )
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Database Build Complete")
        print(f"{'='*60}")
        print(f"Successful: {successful}")
        print(f"No face detected: {no_face}")
        print(f"Failed: {failed}")
        print(f"Total in database: {self.db.get_count()}")
        print(f"{'='*60}\n")