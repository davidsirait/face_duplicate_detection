"""
Face detection module with unified interface
"""
import torch
from facenet_pytorch import MTCNN
import logging
import cv2

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Unified face detector supporting MTCNN and YuNet
    
    Usage:
        detector = FaceDetector(detector_type='mtcnn', device='cpu') 
        face_tensor = detector.detect_and_align(image)
    """
    
    def __init__(self, detector_type='mtcnn', device='cpu', image_size=160, 
                 yunet_model_path='./model/face_detection_yunet_2023mar.onnx'):
        """
        Initialize face detector
        
        Args:
            detector_type: 'mtcnn' or 'yunet'
            device: 'cpu' or 'cuda'
            image_size: Output face size (default 160 for FaceNet)
            yunet_model_path: Path to YuNet ONNX model (only needed if using YuNet)
        """
        self.detector_type = detector_type.lower()
        self.device = device
        self.image_size = image_size
        
        if self.detector_type == 'mtcnn':
            self._init_mtcnn()
        elif self.detector_type == 'yunet':
            self._init_yunet(yunet_model_path)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}. Use 'mtcnn' or 'yunet'")
    
    def _init_mtcnn(self):
        """Initialize MTCNN detector"""
        logger.info("Initializing MTCNN detector")
        self.detector = MTCNN(
            image_size=self.image_size,
            margin=0,
            min_face_size=100, # ignore small faces
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device,
            select_largest=True
        )
    
    def _init_yunet(self, model_path):
        """Initialize YuNet detector"""
        
        logger.info("Initializing YuNet detector")
        try:
            self.detector = cv2.FaceDetectorYN.create(
                model_path,
                "",
                (320, 320),
                score_threshold=0.6,
                nms_threshold=0.3
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load YuNet model from {model_path}: {e}")
    
    def detect_and_align(self, image):
        """
        Detect and align face from image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            torch.Tensor: Aligned face tensor (3, 160, 160) or None if no face detected
        """
        if self.detector_type == 'mtcnn':
            return self._detect_mtcnn(image)
        else:
            return self._detect_yunet(image)
    
    def _detect_mtcnn(self, image):
        """Detect using MTCNN"""
        try:
            face_tensor = self.detector(image)
            return face_tensor
        except Exception as e:
            logger.error(f"MTCNN detection error: {e}")
            return None
    
    def _detect_yunet(self, image):
        """Detect using YuNet"""
        import numpy as np
        
        try:
            # Convert PIL to OpenCV format
            img = np.array(image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            height, width = img.shape[:2]
            self.detector.setInputSize((width, height))
            
            # Detect faces
            _, faces = self.detector.detect(img)
            
            if faces is None or len(faces) == 0:
                return None
            
            # Get largest face
            face = faces[0]
            x, y, w, h = face[:4].astype(int)
            
            # Add margin
            margin = int(0.1 * max(w, h))
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(width - x, w + 2*margin)
            h = min(height - y, h + 2*margin)
            
            # Crop face
            face_img = img[y:y+h, x:x+w]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (self.image_size, self.image_size))
            
            # Convert to tensor and normalize
            face_tensor = torch.FloatTensor(face_img).permute(2, 0, 1) / 255.0
            face_tensor = (face_tensor - 0.5) / 0.5
            
            return face_tensor
            
        except Exception as e:
            logger.error(f"YuNet detection error: {e}")
            return None