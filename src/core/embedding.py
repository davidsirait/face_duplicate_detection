"""
Face embedding extraction using FaceNet
"""
from facenet_pytorch import InceptionResnetV1
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FaceNetEmbedding:
    """FaceNet embedding extractor"""
    
    def __init__(self, model_name='vggface2', device='cpu'):
        """
        Initialize FaceNet model
        
        Args:
            model_name: 'vggface2' or 'casia-webface'
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model = InceptionResnetV1(pretrained=model_name).eval().to(device)
    
    def extract(self, face_tensor):
        """
        Extract embedding from face tensor
        
        Args:
            face_tensor: torch.Tensor (3, 160, 160)
            
        Returns:
            numpy.ndarray: 512-D embedding or None if extraction fails
        """
        if face_tensor is None:
            return None
        
        try:
            with torch.no_grad():
                face_tensor = face_tensor.unsqueeze(0).to(self.device)
                embedding = self.model(face_tensor).cpu().numpy()[0]
            return embedding
        except Exception as e:
            logger.error(f"Embedding extraction error: {e}")
            return None
    
    def extract_batch(self, face_tensors):
        """
        Extract embeddings from batch of face tensors
        
        Args:
            face_tensors: List of torch.Tensor
            
        Returns:
            numpy.ndarray: (N, 512) embeddings
        """
        if not face_tensors:
            return np.array([])
        
        try:
            with torch.no_grad():
                batch = torch.stack(face_tensors).to(self.device)
                embeddings = self.model(batch).cpu().numpy()
            return embeddings
        except Exception as e:
            logger.error(f"Batch embedding extraction error: {e}")
            return np.array([])