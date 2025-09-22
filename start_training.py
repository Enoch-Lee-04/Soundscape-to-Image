"""
Simple Qwen VLM Training Script
Minimal implementation to avoid compatibility issues
"""

import os
import json
import torch
import random
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer
)


class SimpleSafetyDataset:
    """Simple dataset for safety assessment"""
    
    def __init__(self, data, processor, max_length=512):
        self.data = data
        self.processor = processor
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image_path = item['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='black')
        
        # Simple text processing
        text = f"{item['prompt']} {item['expected_response']}"
        
        # Tokenize
        inputs = self.processor.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # For simplicity, use input_ids as labels
        labels = inputs['input_ids'].clone()
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }


def main():
    print("Starting Qwen VLM Safety Assessment Training")
    print("=" * 50)
    
    # Load model
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        trust_remote_code=True
    )
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=None,
    )
    
    print("✓ Model loaded successfully!")
    
    # Load data
    print("Loading training data...")
    with open("qwen_training_data.json", 'r') as f:
        data = json.load(f)
    
    # Use subset for testing
    if len(data) > 10:
        print(f"Using subset of 10 examples for testing")
        data = data[:10]
    
    # Split data
    random.shuffle(data)
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    print(f"Train examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    
    # Create datasets
    train_dataset = SimpleSafetyDataset(train_data, processor)
    val_dataset = SimpleSafetyDataset(val_data, processor)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./qwen_safety_model",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Start training
    print("\nStarting training...")
    print("Note: This is a simplified training setup for testing.")
    print("Training will be slow on CPU but should complete successfully.")
    
    try:
        trainer.train()
        
        # Save model
        print("\nSaving model...")
        trainer.save_model()
        processor.save_pretrained("./qwen_safety_model")
        
        print("✓ Training completed successfully!")
        print("Model saved to: ./qwen_safety_model")
        
    except Exception as e:
        print(f"Training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
