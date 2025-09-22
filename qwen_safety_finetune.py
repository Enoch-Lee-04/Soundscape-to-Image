"""
Qwen VLM Fine-tuning for Street View Safety Assessment
Fine-tune Qwen-VL model on annotated street view images with safety scores
"""

import os
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
from transformers.trainer_utils import get_last_checkpoint
import wandb
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
    use_cache: bool = field(
        default=False,
        metadata={"help": "Whether to use cache for generation"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration"""
    data_path: str = field(
        default="vlm_safety_training_data.json",
        metadata={"help": "Path to training data JSON file"}
    )
    image_folder: str = field(
        default="Sample SVI",
        metadata={"help": "Folder containing street view images"}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    train_split: float = field(
        default=0.8,
        metadata={"help": "Fraction of data to use for training"}
    )


@dataclass
class TrainingArguments:
    """Arguments for training configuration"""
    output_dir: str = field(
        default="./qwen_safety_model",
        metadata={"help": "Output directory for model"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=2,
        metadata={"help": "Training batch size per device"}
    )
    per_device_eval_batch_size: int = field(
        default=2,
        metadata={"help": "Evaluation batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
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
        default=10,
        metadata={"help": "Logging steps"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint steps"}
    )
    eval_steps: int = field(
        default=500,
        metadata={"help": "Evaluation steps"}
    )
    save_total_limit: int = field(
        default=3,
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
        default="wandb",
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
        max_length: int = 2048,
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
            
            # Mask user input (set to -100 so it's not used in loss calculation)
            # This is a simplified approach - in practice, you'd want more sophisticated masking
            # based on the conversation structure
            user_tokens = self.processor.tokenizer.encode(prompt, add_special_tokens=False)
            labels[:len(user_tokens)] = -100
            
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
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set up logging
    if training_args.report_to == "wandb":
        wandb.init(
            project="qwen-safety-assessment",
            name=training_args.run_name,
            config={
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "epochs": training_args.num_train_epochs,
            }
        )
    
    # Load model and processor
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code
    )
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        use_cache=model_args.use_cache
    )
    
    # Load and split data
    print("Loading training data...")
    data = load_training_data(data_args.data_path)
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
        output_dir=training_args.output_dir,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        learning_rate=training_args.learning_rate,
        warmup_ratio=training_args.warmup_ratio,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        eval_steps=training_args.eval_steps,
        save_total_limit=training_args.save_total_limit,
        load_best_model_at_end=training_args.load_best_model_at_end,
        metric_for_best_model=training_args.metric_for_best_model,
        greater_is_better=training_args.greater_is_better,
        report_to=training_args.report_to,
        run_name=training_args.run_name,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.tokenizer,
        data_collator=None,  # We handle collation in the dataset
    )
    
    # Check for checkpoint
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif os.path.isdir(training_args.output_dir):
        checkpoint = get_last_checkpoint(training_args.output_dir)
    
    # Start training
    print("Starting training...")
    trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save final model
    print("Saving final model...")
    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)
    
    # Close wandb
    if training_args.report_to == "wandb":
        wandb.finish()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
