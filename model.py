import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision import models
from config.config import Config

class TextEncoder(nn.Module):
    """Encoder cho text sử dụng PhoBERT"""
    
    def __init__(self, model_name=Config.BERT_MODEL_NAME):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        self.hidden_size = self.bert.config.hidden_size
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.dropout(pooled_output)

class ImageEncoder(nn.Module):
    """Encoder cho image sử dụng ResNet"""
    
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        # Loại bỏ lớp cuối cùng
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        self.hidden_size = 2048
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        return self.dropout(features)

class MultimodalClassifier(nn.Module):
    """Mô hình multimodal kết hợp text và image"""
    
    def __init__(self):
        super(MultimodalClassifier, self).__init__()
        
        # Encoders
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()
        
        # Fusion layer
        self.fusion_dim = 512
        self.text_projection = nn.Linear(self.text_encoder.hidden_size, self.fusion_dim)
        self.image_projection = nn.Linear(self.image_encoder.hidden_size, self.fusion_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(128, Config.NUM_CLASSES)
        )
        
    def forward(self, input_ids, attention_mask, images):
        # Encode text và image
        text_features = self.text_encoder(input_ids, attention_mask)
        image_features = self.image_encoder(images)
        
        # Project to same dimension
        text_proj = self.text_projection(text_features)
        image_proj = self.image_projection(image_features)
        
        # Concatenate features
        combined_features = torch.cat([text_proj, image_proj], dim=1)
        
        # Classify
        logits = self.classifier(combined_features)
        
        return logits

class TextOnlyClassifier(nn.Module):
    """Mô hình chỉ sử dụng text"""
    
    def __init__(self):
        super(TextOnlyClassifier, self).__init__()
        self.text_encoder = TextEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(self.text_encoder.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(128, Config.NUM_CLASSES)
        )
        
    def forward(self, input_ids, attention_mask):
        text_features = self.text_encoder(input_ids, attention_mask)
        logits = self.classifier(text_features)
        return logits

class ImageOnlyClassifier(nn.Module):
    """Mô hình chỉ sử dụng image"""
    
    def __init__(self):
        super(ImageOnlyClassifier, self).__init__()
        self.image_encoder = ImageEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(self.image_encoder.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, Config.NUM_CLASSES)
        )
        
    def forward(self, images):
        image_features = self.image_encoder(images)
        logits = self.classifier(image_features)
        return logits

def create_model(model_type='multimodal'):
    """Factory function để tạo model"""
    if model_type == 'multimodal':
        return MultimodalClassifier()
    elif model_type == 'text_only':
        return TextOnlyClassifier()
    elif model_type == 'image_only':
        return ImageOnlyClassifier()
    else:
        raise ValueError(f"Unknown model type: {model_type}")