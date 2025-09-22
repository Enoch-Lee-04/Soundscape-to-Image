"""
Test the fine-tuned Qwen VLM model for safety assessment
"""

import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def test_finetuned_model():
    """Test the fine-tuned model"""
    print("Testing Fine-tuned Qwen VLM Safety Assessment Model")
    print("=" * 60)
    
    # Load the fine-tuned model
    model_path = "./qwen_safety_model"
    
    if not os.path.exists(model_path):
        print("Error: Fine-tuned model not found!")
        print("Please run training first: python start_training.py")
        return 1
    
    print("Loading fine-tuned model...")
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=None
        )
        print("✓ Fine-tuned model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Test on a sample image
    test_images = [
        "Sample SVI/1.jpg",
        "Sample SVI/2.jpg", 
        "Sample SVI/10.jpg"
    ]
    
    safety_prompts = [
        "Analyze this street view image and provide a safety score from 1-10 (10 being safest). Consider sidewalk condition, traffic density, lighting, and pedestrian safety.",
        "Classify this street view image as SAFE or UNSAFE for pedestrians. Provide your reasoning.",
        "Assess the safety risks in this street view image. Identify any high, medium, or low risk elements."
    ]
    
    print(f"\nTesting on {len(test_images)} sample images...")
    print("=" * 60)
    
    for i, (image_path, prompt) in enumerate(zip(test_images, safety_prompts)):
        if not os.path.exists(image_path):
            print(f"Skipping {image_path} - not found")
            continue
            
        print(f"\nTest {i+1}: {os.path.basename(image_path)}")
        print(f"Prompt: {prompt[:80]}...")
        print("-" * 40)
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Simple text-only inference (since we simplified the training)
            # In a full implementation, you'd use the vision component
            text_input = f"Street view safety assessment: {prompt}"
            
            # Tokenize
            inputs = processor.tokenizer(
                text_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=200,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new generated text
            if text_input in response:
                response = response.split(text_input)[-1].strip()
            
            print(f"Model Response: {response}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    print("\n" + "=" * 60)
    print("✓ Model testing completed!")
    print("=" * 60)
    print("\nNote: This is a simplified test. The model was trained on text-only data")
    print("for this demonstration. In a full implementation, you would:")
    print("1. Use the full vision-language pipeline")
    print("2. Train with proper image-text pairs")
    print("3. Fine-tune the vision encoder as well")
    print("\nYour model is ready for further development and testing!")
    
    return 0


if __name__ == "__main__":
    exit(test_finetuned_model())
