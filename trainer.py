import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from config.config import Config
import os

class ModelTrainer:
    """Class để huấn luyện mô hình"""
    
    def __init__(self, model, device=Config.DEVICE):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.scheduler = StepLR(self.optimizer, step_size=3, gamma=0.1)
        
        # Lưu trữ lịch sử training
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader):
        """Huấn luyện một epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Kiểm tra loại model
            if hasattr(self.model, 'text_encoder') and hasattr(self.model, 'image_encoder'):
                # Multimodal model
                logits = self.model(input_ids, attention_mask, images)
            elif hasattr(self.model, 'text_encoder'):
                # Text only model
                logits = self.model(input_ids, attention_mask)
            else:
                # Image only model
                logits = self.model(images)
            
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """Đánh giá một epoch"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'text_encoder') and hasattr(self.model, 'image_encoder'):
                    logits = self.model(input_ids, attention_mask, images)
                elif hasattr(self.model, 'text_encoder'):
                    logits = self.model(input_ids, attention_mask)
                else:
                    logits = self.model(images)
                
                loss = self.criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def train(self, train_loader, val_loader, num_epochs=Config.EPOCHS):
        """Huấn luyện mô hình"""
        print("Bắt đầu huấn luyện...")
        
        best_val_accuracy = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, val_preds, val_labels = self.validate_epoch(val_loader)
            
            # Scheduler step
            self.scheduler.step()
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_model_state = self.model.state_dict().copy()
                print(f"New best model! Validation accuracy: {val_acc:.4f}")
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        print(f"\nHuấn luyện hoàn thành! Best validation accuracy: {best_val_accuracy:.4f}")
        
        return best_val_accuracy
    
    def evaluate(self, test_loader):
        """Đánh giá mô hình trên test set"""
        print("Đánh giá mô hình trên test set...")
        
        _, accuracy, predictions, labels = self.validate_epoch(test_loader)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Tìm các class thực tế có trong dữ liệu
        unique_labels = sorted(list(set(labels + predictions)))
        target_names = [Config.LABEL_MAPPING[i] for i in unique_labels]
        
        print("\nClassification Report:")
        print(classification_report(labels, predictions, 
                                  labels=unique_labels,
                                  target_names=target_names,
                                  zero_division=0))
        
        # Confusion Matrix - sử dụng tất cả các class có thể có
        cm = confusion_matrix(labels, predictions, labels=list(range(Config.NUM_CLASSES)))
        self.plot_confusion_matrix(cm)
        
        # In phân bố class trong test set
        print(f"\nPhân bố class trong test set:")
        for label in unique_labels:
            count = labels.count(label)
            percentage = count / len(labels) * 100
            print(f"  {Config.LABEL_MAPPING[label]}: {count} mẫu ({percentage:.1f}%)")
        
        return accuracy, predictions, labels
    
    def plot_confusion_matrix(self, cm):
        """Vẽ confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[Config.LABEL_MAPPING[i] for i in range(Config.NUM_CLASSES)],
                   yticklabels=[Config.LABEL_MAPPING[i] for i in range(Config.NUM_CLASSES)])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(Config.OUTPUT_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self):
        """Vẽ lịch sử training"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(Config.OUTPUT_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, path):
        """Lưu mô hình"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, path)
        print(f"Đã lưu mô hình tại: {path}")
    
    def load_model(self, path):
        """Load mô hình"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        print(f"Đã load mô hình từ: {path}")