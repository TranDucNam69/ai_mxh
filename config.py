import torch

class Config:
    # ====== GENERAL ======
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

    # ====== MODEL CONFIG ======
    BERT_MODEL_NAME = "vinai/phobert-base"
    DROPOUT_RATE = 0.3
    NUM_CLASSES = 3
    FUSION_DIM = 512  # Tăng từ 384 lên 512

    # ====== TRAINING CONFIG ======
    MAX_EPOCHS = 20  # Tăng từ mặc định 10 lên 20
    BATCH_SIZE = 32  # Phù hợp với tập dữ liệu 300–500 mẫu
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 1e-4

    # ====== TOKENIZER & DATA ======
    MAX_SEQ_LENGTH = 256

    # ====== PATHS ======
    MODEL_SAVE_PATH = "models/"
    ONNX_EXPORT_PATH = "output/"
    IMAGE_FOLDER = "data/images/"
    CSV_DATA_PATH = "data/posts_data.csv"
    TEXT_MODEL_ONNX = "output/text_model.onnx"
    IMAGE_MODEL_ONNX = "output/image_model.onnx"
    COMBINED_MODEL_ONNX = "output/combined_model.onnx"
