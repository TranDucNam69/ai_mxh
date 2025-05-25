import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
from transformers import AutoTokenizer
from torchvision import transforms
import os
from typing import Dict, List, Tuple
from config.config import Config

class SocialMediaDataset(Dataset):
    """Dataset cho dữ liệu mạng xã hội với text và image"""
    
    def __init__(self, csv_path: str, images_dir: str, tokenizer, transform=None, mode='train'):
        self.data = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.mode = mode
        
        # Lọc bỏ các dòng không có ảnh
        self.data = self.data.dropna(subset=['image_path', 'text', 'label'])
        self.data = self.data.reset_index(drop=True)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Xử lý text
        text = str(row['text'])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=Config.MAX_LENGTH,
            return_tensors='pt'
        )
        
        # Xử lý image
        image_path = os.path.join(self.images_dir, row['image_path'])
        image = self.load_image(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        # Label
        label = int(row['label'])
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def load_image(self, image_path: str):
        """Load và preprocess image"""
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            # Tạo ảnh trắng nếu không load được
            print(f"Error loading image {image_path}: {e}")
            return Image.new('RGB', Config.IMAGE_SIZE, color='white')

class DataProcessor:
    """Xử lý và chuẩn bị dữ liệu"""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        self.image_transform = self.get_image_transform()
    
    def get_image_transform(self):
        """Định nghĩa transform cho image"""
        return transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def create_data_loaders(self, train_ratio=0.8, val_ratio=0.1):
        """Tạo train, validation và test data loaders"""
        
        # Đọc toàn bộ dataset
        full_dataset = SocialMediaDataset(
            Config.CSV_PATH,
            Config.IMAGES_DIR,
            self.tokenizer,
            self.image_transform
        )
        
        # Chia dataset
        dataset_size = len(full_dataset)
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        # Tạo data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, val_loader, test_loader
    
    def analyze_data(self):
        """Phân tích dữ liệu"""
        data = pd.read_csv(Config.CSV_PATH)
        
        print("=== PHÂN TÍCH DỮ LIỆU ===")
        print(f"Tổng số mẫu: {len(data)}")
        print(f"Phân bố nhãn:")
        for label, count in data['label'].value_counts().sort_index().items():
            label_name = Config.LABEL_MAPPING[label]
            print(f"  {label} ({label_name}): {count} mẫu ({count/len(data)*100:.1f}%)")
        
        print(f"\nĐộ dài text trung bình: {data['text'].str.len().mean():.1f} ký tự")
        print(f"Độ dài text tối đa: {data['text'].str.len().max()} ký tự")
        
        return data