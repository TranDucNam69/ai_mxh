import torch
import torch.onnx
from config.config import Config
from models.model import create_model


def export_model_to_onnx(model_type, model=None):
    device = Config.DEVICE
    if model is None:
        model = create_model(model_type)
        model.load_state_dict(torch.load(f"models/{model_type}_model.pth", map_location=device))

    model.to(device)
    model.eval()

    dummy_text_input = {
        'input_ids': torch.randint(0, 1000, (1, Config.MAX_SEQ_LENGTH)).to(device),
        'attention_mask': torch.ones((1, Config.MAX_SEQ_LENGTH)).to(device)
    }
    dummy_image_input = torch.randn(1, 3, 224, 224).to(device)

    if model_type == 'text_only':
        input_names = ["input_ids", "attention_mask"]
        output_names = ["logits"]
        torch.onnx.export(
            model,
            (dummy_text_input['input_ids'], dummy_text_input['attention_mask']),
            Config.TEXT_MODEL_ONNX,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={'input_ids': {0: 'batch'}, 'attention_mask': {0: 'batch'}, 'logits': {0: 'batch'}},
            opset_version=11
        )
    elif model_type == 'image_only':
        input_names = ["images"]
        output_names = ["logits"]
        torch.onnx.export(
            model,
            (dummy_image_input,),
            Config.IMAGE_MODEL_ONNX,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={'images': {0: 'batch'}, 'logits': {0: 'batch'}},
            opset_version=11
        )
    else:
        input_names = ["input_ids", "attention_mask", "images"]
        output_names = ["logits"]
        torch.onnx.export(
            model,
            (dummy_text_input['input_ids'], dummy_text_input['attention_mask'], dummy_image_input),
            Config.COMBINED_MODEL_ONNX,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                'input_ids': {0: 'batch'},
                'attention_mask': {0: 'batch'},
                'images': {0: 'batch'},
                'logits': {0: 'batch'}
            },
            opset_version=11
        )

    print(f"Exported {model_type} model to ONNX successfully.")
