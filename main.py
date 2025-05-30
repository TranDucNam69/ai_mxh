import argparse
import torch
from config.config import Config
from models.model import create_model
from training.trainer import train
from export.onnx_exporter import export_model_to_onnx


def main():
    parser = argparse.ArgumentParser(description="Social Media Classifier")
    parser.add_argument("--mode", type=str, choices=["train", "export"], default="train")
    parser.add_argument("--model_type", type=str, choices=["multimodal", "text_only", "image_only"], default="multimodal")
    args = parser.parse_args()

    model = create_model(model_type=args.model_type)
    model = model.to(Config.DEVICE)

    if args.mode == "train":
        print("=== BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH ===")
        trained_model = train(model, args.model_type)

        print("\n=== XUẤT MÔ HÌNH ONNX SAU HUẤN LUYỆN ===")
        export_model_to_onnx(trained_model, args.model_type)

    elif args.mode == "export":
        print("=== XUẤT MÔ HÌNH ĐÃ HUẤN LUYỆN SANG ONNX ===")
        export_model_to_onnx(model, args.model_type)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")  # Tăng tốc inference trên GPU A100/RTX
    main()
