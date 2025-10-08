"""
Vector database management using ChromaDB
"""
import chromadb
from chromadb.config import Settings
import uuid
import numpy as np
from typing import List, Dict, Optional, Tuple
import base64
from io import BytesIO
from PIL import Image


class FaceVectorDB:
    """ChromaDB wrapper for face embeddings storage and similarity search"""
    
    def __init__(self, persist_directory="./face_db", collection_name="face_embeddings"):
        """
        Initialize ChromaDB client
        
        Args:
            persist_directory: Directory to persist database
            collection_name: Name of the collection
        """
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Face embeddings for duplicate detection"}
        )
    
    def _create_thumbnail_base64(self, image_path, size=(100, 100)):
        """
        Create thumbnail and convert to base64 string
        
        Args:
            image_path: Path to original image
            size: Thumbnail size (default 100x100)
            
        Returns:
            str: Base64 encoded thumbnail
        """
        try:
            img = Image.open(image_path)
            img.thumbnail(size, Image.LANCZOS)
            
            # Convert to base64
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return img_base64
        except Exception as e:
            print(f"Error creating thumbnail: {e}")
            return None
    
    def add_embedding(self, embedding: np.ndarray, image_path: str, 
                     person_id: Optional[str] = None, metadata: Optional[Dict] = None):
        """Add embedding with thumbnail"""
        if metadata is None:
            metadata = {}
        
        # Create and add thumbnail
        thumbnail_b64 = self._create_thumbnail_base64(image_path)
        
        metadata.update({
            "image_path": str(image_path),
            "person_id": person_id or "unknown",
            "thumbnail": thumbnail_b64  # ← Store thumbnail here
        })
        
        doc_id = str(uuid.uuid4())
        
        self.collection.add(
            embeddings=[embedding.tolist()],
            documents=[str(image_path)],
            ids=[doc_id],
            metadatas=[metadata]
        )
        
        return doc_id
    
    def add_embeddings_batch(self, embeddings: List[np.ndarray], 
                            image_paths: List[str],
                            person_ids: Optional[List[str]] = None,
                            metadatas: Optional[List[Dict]] = None):
        """Add batch with thumbnails"""
        if person_ids is None:
            person_ids = ["unknown"] * len(embeddings)
        
        if metadatas is None:
            metadatas = [{}] * len(embeddings)
        
        ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
        
        # Add thumbnails to metadata
        for i, (metadata, img_path) in enumerate(zip(metadatas, image_paths)):
            thumbnail_b64 = self._create_thumbnail_base64(img_path)
            metadata.update({
                "image_path": str(img_path),
                "person_id": person_ids[i],
                "thumbnail": thumbnail_b64
            })
        
        self.collection.add(
            embeddings=[emb.tolist() for emb in embeddings],
            documents=[str(path) for path in image_paths],
            ids=ids,
            metadatas=metadatas
        )
        
        return ids

    
    def search(self, embedding: np.ndarray, n_results: int = 5) -> Dict:
        """
        Search for similar faces
        
        Args:
            embedding: Query face embedding
            n_results: Number of results to return
            
        Returns:
            Dict: Search results with distances, documents, and metadata
        """
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=n_results
        )
        return results
    
    def check_duplicate(self, embedding: np.ndarray, 
                       threshold: float = 0.6) -> Tuple[bool, Optional[Dict]]:
        """
        Check if face is a duplicate
        
        Args:
            embedding: Query face embedding
            threshold: Distance threshold for duplicate (lower = stricter)
            
        Returns:
            Tuple[bool, Optional[Dict]]: (is_duplicate, match_info)
        """
        results = self.search(embedding, n_results=1)
        
        if not results['distances'][0]:
            return False, None
        
        distance = results['distances'][0][0]
        
        if distance < threshold:
            return True, {
                'image_path': results['documents'][0][0],
                'metadata': results['metadatas'][0][0],
                'distance': distance,
                'id': results['ids'][0][0]
            }
        
        return False, {'distance': distance}
    
    def get_count(self) -> int:
        """Get total number of embeddings in database"""
        return self.collection.count()
    
    def clear(self):
        """Clear all embeddings from database"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"description": "Face embeddings for duplicate detection"}
        )