"""
Simplified Qwen VLM Fine-tuning for Street View Safety Assessment
Works without DeepSpeed and other optional dependencies
"""

import os
import sys
import json
import torch
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    HfArgumentParser
)
from tqdm import tqdm


@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    model_name_or_path: str = field(
        default="Qwen/Qwen2-VL-2B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading model"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration"""
    data_path: str = field(
        default="qwen_training_data.json",
        metadata={"help": "Path to training data JSON file"}
    )
    image_folder: str = field(
        default="Sample SVI",
        metadata={"help": "Folder containing street view images"}
    )
    max_length: int = field(
        default=1024,  # Reduced for CPU training
        metadata={"help": "Maximum sequence length"}
    )
    train_split: float = field(
        default=0.8,
        metadata={"help": "Fraction of data to use for training"}
    )


@dataclass
class CustomTrainingArguments:
    """Arguments for training configuration"""
    output_dir: str = field(
        default="./qwen_safety_model",
        metadata={"help": "Output directory for model"}
    )
    num_train_epochs: int = field(
        default=2,  # Reduced for testing
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=1,  # CPU-friendly batch size
        metadata={"help": "Training batch size per device"}
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": "Evaluation batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=8,  # Increased to maintain effective batch size
        metadata={"help": "Number of steps to accumulate gradients"}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "Learning rate"}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Warmup ratio"}
    )
    logging_steps: int = field(
        default=5,
        metadata={"help": "Logging steps"}
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Save checkpoint steps"}
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Evaluation steps"}
    )
    save_total_limit: int = field(
        default=2,
        metadata={"help": "Maximum number of checkpoints to save"}
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "Load best model at end"}
    )
    metric_for_best_model: str = field(
        default="eval_loss",
        metadata={"help": "Metric for best model selection"}
    )
    greater_is_better: bool = field(
        default=False,
        metadata={"help": "Whether greater metric is better"}
    )
    report_to: str = field(
        default="none",  # Disable wandb for simplicity
        metadata={"help": "Reporting tool (wandb, tensorboard, none)"}
    )
    run_name: str = field(
        default="qwen_safety_assessment",
        metadata={"help": "Run name for logging"}
    )


class SafetyDataset(Dataset):
    """Dataset for safety assessment fine-tuning"""
    
    def __init__(
        self, 
        data: List[Dict], 
        processor: AutoProcessor,
        max_length: int = 1024,
        is_training: bool = True
    ):
        self.data = data
        self.processor = processor
        self.max_length = max_length
        self.is_training = is_training
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image_path = item['image_path']
        if not os.path.exists(image_path):
            # Try relative path
            image_path = os.path.join("Sample SVI", os.path.basename(image_path))
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Create a dummy image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Prepare text
        prompt = item['prompt']
        expected_response = item['expected_response']
        
        # Create conversation format for Qwen-VL
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            },
            {
                "role": "assistant", 
                "content": expected_response
            }
        ]
        
        # Process with tokenizer
        processed = self.processor(
            conversation=conversation,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Prepare labels for training
        if self.is_training:
            # Create labels for the assistant's response
            input_ids = processed['input_ids'].squeeze(0)
            labels = input_ids.clone()
            
            # Simple masking: mask first half (user input)
            mask_length = len(input_ids) // 2
            labels[:mask_length] = -100
            
            processed['labels'] = labels.unsqueeze(0)
        
        return {
            'input_ids': processed['input_ids'].squeeze(0),
            'attention_mask': processed['attention_mask'].squeeze(0),
            'pixel_values': processed['pixel_values'].squeeze(0),
            'labels': processed.get('labels', processed['input_ids']).squeeze(0) if self.is_training else processed['input_ids'].squeeze(0)
        }


def load_training_data(data_path: str) -> List[Dict]:
    """Load and preprocess training data"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} training examples")
    return data


def split_data(data: List[Dict], train_split: float = 0.8) -> tuple:
    """Split data into train and validation sets"""
    random.shuffle(data)
    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Train examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    
    return train_data, val_data


def main():
    print("Qwen VLM Safety Assessment Fine-tuning (Simplified)")
    print("=" * 60)
    
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, custom_args = parser.parse_args_into_dataclasses()
    
    # Load model and processor
    print("Loading model and processor...")
    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code
        )
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map=None,  # No device mapping for CPU
        )
        
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("This might be due to network issues or model access permissions.")
        print("Please ensure you have internet access and try again.")
        return 1
    
    # Load and split data
    print("Loading training data...")
    if not os.path.exists(data_args.data_path):
        print(f"Error: Training data file {data_args.data_path} not found!")
        print("Please run convert_data_for_qwen.py first.")
        return 1
    
    data = load_training_data(data_args.data_path)
    
    # Use only a subset for testing
    if len(data) > 20:
        print(f"Using subset of {min(20, len(data))} examples for testing")
        data = data[:20]
    
    train_data, val_data = split_data(data, data_args.train_split)
    
    # Create datasets
    train_dataset = SafetyDataset(
        train_data, 
        processor, 
        data_args.max_length, 
        is_training=True
    )
    val_dataset = SafetyDataset(
        val_data, 
        processor, 
        data_args.max_length, 
        is_training=False
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=custom_args.output_dir,
        num_train_epochs=custom_args.num_train_epochs,
        per_device_train_batch_size=custom_args.per_device_train_batch_size,
        per_device_eval_batch_size=custom_args.per_device_eval_batch_size,
        gradient_accumulation_steps=custom_args.gradient_accumulation_steps,
        learning_rate=custom_args.learning_rate,
        warmup_ratio=custom_args.warmup_ratio,
        logging_steps=custom_args.logging_steps,
        save_steps=custom_args.save_steps,
        eval_steps=custom_args.eval_steps,
        save_total_limit=custom_args.save_total_limit,
        load_best_model_at_end=custom_args.load_best_model_at_end,
        metric_for_best_model=custom_args.metric_for_best_model,
        greater_is_better=custom_args.greater_is_better,
        report_to=custom_args.report_to,
        run_name=custom_args.run_name,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        data_collator=None,  # We handle collation in the dataset
    )
    
    # Check for existing model directory
    if os.path.isdir(training_args.output_dir):
        print(f"Warning: Output directory {training_args.output_dir} already exists. Training will overwrite existing model.")

    # Start training
    print("Starting training...")
    print(f"Training on {len(train_dataset)} examples")
    print(f"Validating on {len(val_dataset)} examples")
    print("Note: CPU training will be slow. Consider using a GPU for faster training.")
    
    try:
        trainer.train()
        
        # Save final model
        print("Saving final model...")
        trainer.save_model()
        processor.save_pretrained(training_args.output_dir)
        
        print("Training completed successfully!")
        print(f"Model saved to: {training_args.output_dir}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
