#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hệ thống AI đánh giá bài đăng mạng xã hội
Tác giả: AI Assistant
Mô tả: Hệ thống multimodal để phân loại bài đăng thành 3 loại: Lành mạnh, Gây tranh cãi, Độc hại
"""

import torch
import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.config import Config
from data.data_loader import DataProcessor
from models.model import create_model
from training.trainer import ModelTrainer
from export.onnx_exporter import ONNXExporter, ONNXPredictor

def setup_environment():
    """Thiết lập môi trường"""
    # Tạo các thư mục cần thiết
    Config.create_dirs()
    
    # Kiểm tra CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available - Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠ CUDA not available - Using CPU")
    
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ Device: {Config.DEVICE}")

def train_models():
    """Huấn luyện các mô hình"""
    print("=== BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH ===")
    
    # Khởi tạo data processor
    data_processor = DataProcessor()
    
    # Phân tích dữ liệu
    data_processor.analyze_data()
    
    # Tạo data loaders
    train_loader, val_loader, test_loader = data_processor.create_data_loaders()
    
    results = {}
    
    # 1. Huấn luyện Multimodal Model
    print("\n1. HUẤN LUYỆN MULTIMODAL MODEL")
    print("=" * 50)
    multimodal_model = create_model('multimodal')
    multimodal_trainer = ModelTrainer(multimodal_model)
    
    multimodal_acc = multimodal_trainer.train(train_loader, val_loader)
    test_acc, _, _ = multimodal_trainer.evaluate(test_loader)
    results['multimodal'] = {'val_acc': multimodal_acc, 'test_acc': test_acc}
    
    # Lưu model
    multimodal_trainer.save_model(Config.MODEL_DIR / 'multimodal_model.pth')
    multimodal_trainer.plot_training_history()
    
    # 2. Huấn luyện Text-only Model
    print("\n2. HUẤN LUYỆN TEXT-ONLY MODEL")
    print("=" * 50)
    text_model = create_model('text_only')
    text_trainer = ModelTrainer(text_model)
    
    text_acc = text_trainer.train(train_loader, val_loader)
    test_acc, _, _ = text_trainer.evaluate(test_loader)
    results['text_only'] = {'val_acc': text_acc, 'test_acc': test_acc}
    
    text_trainer.save_model(Config.MODEL_DIR / 'text_model.pth')
    
    # 3. Huấn luyện Image-only Model
    print("\n3. HUẤN LUYỆN IMAGE-ONLY MODEL")
    print("=" * 50)
    image_model = create_model('image_only')
    image_trainer = ModelTrainer(image_model)
    
    image_acc = image_trainer.train(train_loader, val_loader)
    test_acc, _, _ = image_trainer.evaluate(test_loader)
    results['image_only'] = {'val_acc': image_acc, 'test_acc': test_acc}
    
    image_trainer.save_model(Config.MODEL_DIR / 'image_model.pth')
    
    # So sánh kết quả
    print("\n=== TỔNG KẾT KẾT QUẢ ===")
    for model_name, metrics in results.items():
        print(f"{model_name.upper()}:")
        print(f"  Validation Accuracy: {metrics['val_acc']:.4f}")
        print(f"  Test Accuracy: {metrics['test_acc']:.4f}")
    
    return multimodal_trainer.model, text_trainer.model, image_trainer.model

def export_to_onnx(multimodal_model=None, text_model=None, image_model=None):
    """Xuất mô hình sang ONNX"""
    print("\n=== XUẤT MÔ HÌNH SANG ONNX ===")
    
    exporter = ONNXExporter()
    exporter.export_all_models(multimodal_model, text_model, image_model)

def test_onnx_inference():
    """Test inference với ONNX model"""
    print("\n=== TEST ONNX INFERENCE ===")
    
    if Config.ONNX_COMBINED_MODEL_PATH.exists():
        predictor = ONNXPredictor(str(Config.ONNX_COMBINED_MODEL_PATH), 'multimodal')
        
        # Test với text và image mẫu
        sample_text = "Đây là một bài đăng tích cực và lành mạnh"
        
        # Tạo ảnh mẫu nếu không có
        sample_image_path = Config.DATA_DIR / "sample_image.jpg"
        if not sample_image_path.exists():
            from PIL import Image
            sample_image = Image.new('RGB', Config.IMAGE_SIZE, color='white')
            sample_image.save(sample_image_path)
        
        try:
            result = predictor.predict(text=sample_text, image_path=str(sample_image_path))
            print("✓ ONNX inference test successful!")
            print(f"  Predicted: {result['predicted_label']}")
            print(f"  Confidence: {result['confidence']:.4f}")
        except Exception as e:
            print(f"✗ ONNX inference test failed: {e}")

def create_sample_data():
    """Tạo dữ liệu mẫu để test - đảm bảo có đủ cả 3 class"""
    print("\n=== TẠO DỮ LIỆU MẪU ===")
    
    import pandas as pd
    from PIL import Image
    import numpy as np
    
    # Tạo dữ liệu cân bằng hơn - mỗi class có ít nhất 10 samples
    sample_data = {
        'text': [
            # Class 0: Lành mạnh (10 samples)
            'Hôm nay thật là một ngày tuyệt vời!',
            'Cảm ơn mọi người đã chia sẻ',
            'Chúc mọi người một ngày tốt lành',
            'Rất vui được gặp gỡ mọi người',
            'Tôi yêu gia đình mình',
            'Học tập chăm chỉ để có tương lai tốt',
            'Cùng nhau xây dựng cộng đồng tích cực',
            'Hãy luôn giữ tinh thần lạc quan',
            'Chia sẻ kiến thức là điều tuyệt vời',
            'Mọi người đều có quyền được hạnh phúc',
            
            # Class 1: Gây tranh cãi (10 samples)
            'Tôi không đồng ý với quan điểm này',
            'Bài viết này gây tranh cãi nhưng có thể thảo luận',
            'Có lẽ chúng ta nên xem xét kỹ hơn',
            'Điều này có thể gây hiểu lầm',
            'Quan điểm này cần được thảo luận thêm',
            'Tôi có ý kiến khác về vấn đề này',
            'Vấn đề này phức tạp hơn chúng ta nghĩ',
            'Cần có thêm bằng chứng để chứng minh',
            'Nhiều người có quan điểm khác nhau',
            'Đây là chủ đề nhạy cảm cần cân nhắc',
            
            # Class 2: Độc hại (10 samples)
            'Đây là nội dung không phù hợp và độc hại',
            'Tôi ghét tất cả mọi thứ',
            'Thế giới này thật tệ hại',
            'Không ai hiểu tôi cả',
            'Tôi muốn làm hại người khác',
            'Mọi người đều xấu và ác',
            'Tôi không thể chịu đựng được nữa',
            'Hãy làm tổn thương những kẻ thù',
            'Bạo lực là giải pháp duy nhất',
            'Tôi muốn phá hủy mọi thứ'
        ],
        'image_path': [],
        'label': []
    }
    
    # Tạo image paths và labels tương ứng
    for i in range(30):  # 30 samples total
        sample_data['image_path'].append(f'sample_{i+1:02d}.jpg')
        if i < 10:
            sample_data['label'].append(0)  # Lành mạnh
        elif i < 20:
            sample_data['label'].append(1)  # Gây tranh cãi
        else:
            sample_data['label'].append(2)  # Độc hại
    
    # Trộn dữ liệu để tạo sự ngẫu nhiên
    indices = list(range(30))
    np.random.seed(42)  # Để có thể reproduce
    np.random.shuffle(indices)
    
    shuffled_data = {
        'text': [sample_data['text'][i] for i in indices],
        'image_path': [sample_data['image_path'][i] for i in indices],
        'label': [sample_data['label'][i] for i in indices]
    }
    
    df = pd.DataFrame(shuffled_data)
    df.to_csv(Config.CSV_PATH, index=False, encoding='utf-8')
    
    # Tạo ảnh mẫu với màu sắc khác nhau cho mỗi class
    Config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    for i in range(30):
        # Màu sắc tương ứng với class
        original_idx = indices[i]
        if original_idx < 10:
            color = 'lightgreen'  # Lành mạnh - xanh lá
        elif original_idx < 20:
            color = 'lightyellow'  # Gây tranh cãi - vàng
        else:
            color = 'lightcoral'  # Độc hại - đỏ
            
        image = Image.new('RGB', Config.IMAGE_SIZE, color=color)
        image.save(Config.IMAGES_DIR / f'sample_{i+1:02d}.jpg')
    
    print(f"✓ Đã tạo dữ liệu mẫu tại {Config.CSV_PATH}")
    print(f"✓ Đã tạo 30 ảnh mẫu tại {Config.IMAGES_DIR}")
    print(f"✓ Phân bố: 10 mẫu cho mỗi class (Lành mạnh, Gây tranh cãi, Độc hại)")

def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(description='Social Media Post Classifier')
    parser.add_argument('--mode', choices=['train', 'export', 'test', 'create_sample', 'all'], 
                       default='all', help='Mode to run')
    parser.add_argument('--model_type', choices=['multimodal', 'text_only', 'image_only'], 
                       default='multimodal', help='Model type to use')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS, help='Number of epochs')
    
    args = parser.parse_args()
    
    # Thiết lập môi trường
    setup_environment()
    
    # Cập nhật config nếu cần
    if args.epochs != Config.EPOCHS:
        Config.EPOCHS = args.epochs
    
    try:
        if args.mode == 'create_sample':
            create_sample_data()
            
        elif args.mode == 'train':
            if not Config.CSV_PATH.exists():
                print("⚠ Không tìm thấy file dữ liệu. Tạo dữ liệu mẫu...")
                create_sample_data()
            
            multimodal_model, text_model, image_model = train_models()
            
        elif args.mode == 'export':
            # Load models và export
            multimodal_model = create_model('multimodal')
            text_model = create_model('text_only')
            image_model = create_model('image_only')
            
            # Load weights nếu có
            if (Config.MODEL_DIR / 'multimodal_model.pth').exists():
                checkpoint = torch.load(Config.MODEL_DIR / 'multimodal_model.pth')
                multimodal_model.load_state_dict(checkpoint['model_state_dict'])
            
            export_to_onnx(multimodal_model, text_model, image_model)
            
        elif args.mode == 'test':
            test_onnx_inference()
            
        elif args.mode == 'all':
            # Chạy toàn bộ pipeline
            if not Config.CSV_PATH.exists():
                create_sample_data()
            
            multimodal_model, text_model, image_model = train_models()
            export_to_onnx(multimodal_model, text_model, image_model)
            test_onnx_inference()
            
        print("\n✓ Hoàn thành thành công!")
        
    except KeyboardInterrupt:
        print("\n⚠ Chương trình bị dừng bởi người dùng")
    except Exception as e:
        print(f"\n✗ Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()