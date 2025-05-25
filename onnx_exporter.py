import torch
import torch.onnx
import onnx
import onnxruntime
import numpy as np
from config.config import Config
from models.model import MultimodalClassifier, TextOnlyClassifier, ImageOnlyClassifier
from transformers import AutoTokenizer
from PIL import Image
import torchvision.transforms as transforms

class ONNXExporter:
    """Class để xuất mô hình sang ONNX"""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        self.image_transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def export_multimodal_model(self, model, output_path):
        """Xuất multimodal model sang ONNX"""
        model.eval()
        
        # Tạo dummy inputs
        batch_size = 1
        dummy_input_ids = torch.randint(0, 1000, (batch_size, Config.MAX_LENGTH))
        dummy_attention_mask = torch.ones(batch_size, Config.MAX_LENGTH)
        dummy_images = torch.randn(batch_size, 3, *Config.IMAGE_SIZE)
        
        # Dynamic axes để support batch size khác nhau
        dynamic_axes = {
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'images': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        
        # Export to ONNX
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask, dummy_images),
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask', 'images'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        
        print(f"Đã xuất multimodal model sang ONNX: {output_path}")
        
        # Verify ONNX model
        self.verify_onnx_model(output_path, 
                              [dummy_input_ids.numpy(), 
                               dummy_attention_mask.numpy(), 
                               dummy_images.numpy()])
    
    def export_text_model(self, model, output_path):
        """Xuất text-only model sang ONNX"""
        model.eval()
        
        # Tạo dummy inputs
        batch_size = 1
        dummy_input_ids = torch.randint(0, 1000, (batch_size, Config.MAX_LENGTH))
        dummy_attention_mask = torch.ones(batch_size, Config.MAX_LENGTH)
        
        dynamic_axes = {
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        
        print(f"Đã xuất text model sang ONNX: {output_path}")
        
        self.verify_onnx_model(output_path, 
                              [dummy_input_ids.numpy(), 
                               dummy_attention_mask.numpy()])
    
    def export_image_model(self, model, output_path):
        """Xuất image-only model sang ONNX"""
        model.eval()
        
        # Tạo dummy inputs
        batch_size = 1
        dummy_images = torch.randn(batch_size, 3, *Config.IMAGE_SIZE)
        
        dynamic_axes = {
            'images': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        
        torch.onnx.export(
            model,
            dummy_images,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        
        print(f"Đã xuất image model sang ONNX: {output_path}")
        
        self.verify_onnx_model(output_path, [dummy_images.numpy()])
    
    def verify_onnx_model(self, onnx_path, dummy_inputs):
        """Kiểm tra ONNX model"""
        try:
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Test với ONNX Runtime
            ort_session = onnxruntime.InferenceSession(onnx_path)
            
            # Tạo input dict
            input_names = [input.name for input in ort_session.get_inputs()]
            ort_inputs = {name: inp for name, inp in zip(input_names, dummy_inputs)}
            
            # Run inference
            ort_outputs = ort_session.run(None, ort_inputs)
            
            print(f"✓ ONNX model verification successful!")
            print(f"  Input shapes: {[inp.shape for inp in dummy_inputs]}")
            print(f"  Output shape: {ort_outputs[0].shape}")
            
        except Exception as e:
            print(f"✗ ONNX model verification failed: {e}")
    
    def create_onnx_inference_example(self, onnx_path, model_type='multimodal'):
        """Tạo ví dụ inference với ONNX"""
        
        example_code = f"""
# Ví dụ sử dụng ONNX model trong C#
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

public class SocialMediaClassifier
{{
    private InferenceSession session;
    private readonly Dictionary<int, string> labelMapping = new Dictionary<int, string>
    {{
        {{0, "Lành mạnh"}},
        {{1, "Gây tranh cãi"}},
        {{2, "Độc hại"}}
    }};
    
    public SocialMediaClassifier(string modelPath)
    {{
        session = new InferenceSession(modelPath);
    }}
    
    public (string label, float confidence) Predict("""
        
        if model_type == 'multimodal':
            example_code += """int[] inputIds, int[] attentionMask, float[] imageData)
    {
        // Tạo input tensors
        var inputIdsTensor = new DenseTensor<int>(inputIds, new[] { 1, inputIds.Length });
        var attentionMaskTensor = new DenseTensor<int>(attentionMask, new[] { 1, attentionMask.Length });
        var imagesTensor = new DenseTensor<float>(imageData, new[] { 1, 3, 224, 224 });
        
        // Tạo input dictionary
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor),
            NamedOnnxValue.CreateFromTensor("images", imagesTensor)
        };"""
        elif model_type == 'text_only':
            example_code += """int[] inputIds, int[] attentionMask)
    {
        // Tạo input tensors
        var inputIdsTensor = new DenseTensor<int>(inputIds, new[] { 1, inputIds.Length });
        var attentionMaskTensor = new DenseTensor<int>(attentionMask, new[] { 1, attentionMask.Length });
        
        // Tạo input dictionary
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor)
        };"""
        else:  # image_only
            example_code += """float[] imageData)
    {
        // Tạo input tensor
        var imagesTensor = new DenseTensor<float>(imageData, new[] { 1, 3, 224, 224 });
        
        // Tạo input dictionary
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("images", imagesTensor)
        };"""
        
        example_code += """
        
        // Run inference
        using var results = session.Run(inputs);
        var output = results.First().AsTensor<float>().ToArray();
        
        // Get prediction
        int predictedClass = Array.IndexOf(output, output.Max());
        float confidence = Softmax(output)[predictedClass];
        
        return (labelMapping[predictedClass], confidence);
    }
    
    private float[] Softmax(float[] values)
    {
        var max = values.Max();
        var exp = values.Select(v => Math.Exp(v - max)).ToArray();
        var sum = exp.Sum();
        return exp.Select(e => (float)(e / sum)).ToArray();
    }
    
    public void Dispose()
    {
        session?.Dispose();
    }
}
"""
        
        # Lưu example code
        example_path = Config.OUTPUT_DIR / f"{model_type}_csharp_example.cs"
        with open(example_path, 'w', encoding='utf-8') as f:
            f.write(example_code)
        
        print(f"Đã tạo ví dụ C# tại: {example_path}")
    
    def export_all_models(self, multimodal_model=None, text_model=None, image_model=None):
        """Xuất tất cả các loại model"""
        Config.create_dirs()
        
        if multimodal_model is not None:
            self.export_multimodal_model(
                multimodal_model, 
                Config.ONNX_COMBINED_MODEL_PATH
            )
            self.create_onnx_inference_example(
                Config.ONNX_COMBINED_MODEL_PATH, 
                'multimodal'
            )
        
        if text_model is not None:
            self.export_text_model(
                text_model, 
                Config.ONNX_TEXT_MODEL_PATH
            )
            self.create_onnx_inference_example(
                Config.ONNX_TEXT_MODEL_PATH, 
                'text_only'
            )
        
        if image_model is not None:
            self.export_image_model(
                image_model, 
                Config.ONNX_IMAGE_MODEL_PATH
            )
            self.create_onnx_inference_example(
                Config.ONNX_IMAGE_MODEL_PATH, 
                'image_only'
            )
        
        print("Hoàn thành xuất tất cả ONNX models!")

class ONNXPredictor:
    """Class để thực hiện inference với ONNX model"""
    
    def __init__(self, onnx_path, model_type='multimodal'):
        self.session = onnxruntime.InferenceSession(onnx_path)
        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        self.image_transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_text(self, text):
        """Tiền xử lý text"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=Config.MAX_LENGTH,
            return_tensors='np'
        )
        return encoding['input_ids'], encoding['attention_mask']
    
    def preprocess_image(self, image_path):
        """Tiền xử lý image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_transform(image)
        return image_tensor.unsqueeze(0).numpy()
    
    def predict(self, text=None, image_path=None):
        """Thực hiện dự đoán"""
        inputs = {}
        
        if self.model_type in ['multimodal', 'text_only'] and text is not None:
            input_ids, attention_mask = self.preprocess_text(text)
            inputs['input_ids'] = input_ids
            inputs['attention_mask'] = attention_mask
        
        if self.model_type in ['multimodal', 'image_only'] and image_path is not None:
            image_data = self.preprocess_image(image_path)
            inputs['images'] = image_data
        
        # Run inference
        outputs = self.session.run(None, inputs)
        logits = outputs[0][0]
        
        # Softmax để tính probability
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        return {
            'predicted_class': int(predicted_class),
            'predicted_label': Config.LABEL_MAPPING[predicted_class],
            'confidence': float(confidence),
            'probabilities': {
                Config.LABEL_MAPPING[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
        }