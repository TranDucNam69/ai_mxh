import os
import torch
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    OUTPUT_DIR = BASE_DIR / "output"
    
    # Data paths
    CSV_PATH = DATA_DIR / "posts_data.csv"
    IMAGES_DIR = DATA_DIR / "images"
    
    # Model paths
    BERT_MODEL_NAME = "vinai/phobert-base"
    RESNET_MODEL_NAME = "resnet50"
    
    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 10
    MAX_LENGTH = 256
    
    # Model parameters
    NUM_CLASSES = 3
    DROPOUT_RATE = 0.3
    
    # Image parameters
    IMAGE_SIZE = (224, 224)
    
    # ONNX export
    ONNX_TEXT_MODEL_PATH = OUTPUT_DIR / "text_model.onnx" 
    ONNX_IMAGE_MODEL_PATH = OUTPUT_DIR / "image_model.onnx"
    ONNX_COMBINED_MODEL_PATH = OUTPUT_DIR / "combined_model.onnx"
    
    # Labels
    LABEL_MAPPING = {
        0: "Lành mạnh",
        1: "Gây tranh cãi", 
        2: "Độc hại"
    }
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def create_dirs(cls):
        """Tạo các thư mục cần thiết"""
        for dir_path in [cls.DATA_DIR, cls.MODEL_DIR, cls.OUTPUT_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)