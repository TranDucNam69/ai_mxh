import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision import models
from config.config import Config

class TextEncoder(nn.Module):
    def __init__(self, model_name=Config.BERT_MODEL_NAME):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        self.layernorm = nn.LayerNorm(self.bert.config.hidden_size)
        self.hidden_size = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.layernorm(pooled_output)
        return self.dropout(pooled_output)

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(base_model.children())[:-1])
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        self.layernorm = nn.LayerNorm(2048)
        self.hidden_size = 2048

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.layernorm(features)
        return self.dropout(features)

class MultimodalClassifier(nn.Module):
    def __init__(self):
        super(MultimodalClassifier, self).__init__()
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()

        self.fusion_dim = 384
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_encoder.hidden_size, self.fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(self.fusion_dim)
        )
        self.image_projection = nn.Sequential(
            nn.Linear(self.image_encoder.hidden_size, self.fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(self.fusion_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim * 2, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(96, Config.NUM_CLASSES)
        )

    def forward(self, input_ids, attention_mask, images):
        text_features = self.text_encoder(input_ids, attention_mask)
        image_features = self.image_encoder(images)

        text_proj = self.text_projection(text_features)
        image_proj = self.image_projection(image_features)
        combined = torch.cat([text_proj, image_proj], dim=1)
        logits = self.classifier(combined)
        return logits

class TextOnlyClassifier(nn.Module):
    def __init__(self):
        super(TextOnlyClassifier, self).__init__()
        self.encoder = TextEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(128, Config.NUM_CLASSES)
        )

    def forward(self, input_ids, attention_mask):
        features = self.encoder(input_ids, attention_mask)
        return self.classifier(features)

class ImageOnlyClassifier(nn.Module):
    def __init__(self):
        super(ImageOnlyClassifier, self).__init__()
        self.encoder = ImageEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.hidden_size, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(384, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(128, Config.NUM_CLASSES)
        )

    def forward(self, images):
        features = self.encoder(images)
        return self.classifier(features)

def create_model(model_type='multimodal'):
    if model_type == 'multimodal':
        return MultimodalClassifier()
    elif model_type == 'text_only':
        return TextOnlyClassifier()
    elif model_type == 'image_only':
        return ImageOnlyClassifier()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
