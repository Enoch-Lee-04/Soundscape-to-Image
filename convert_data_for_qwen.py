"""
Convert VLM Safety Training Data for Qwen Fine-tuning
Ensures data format compatibility and validates image paths
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import random


def validate_image_paths(data: List[Dict], image_folder: str = "Sample SVI") -> List[Dict]:
    """Validate and fix image paths in the training data"""
    valid_data = []
    invalid_count = 0
    
    print("Validating image paths...")
    
    for i, item in enumerate(data):
        image_path = item['image_path']
        
        # Try different path variations
        possible_paths = [
            image_path,  # Original path
            os.path.join(image_folder, os.path.basename(image_path)),  # Relative path
            os.path.join(os.getcwd(), image_folder, os.path.basename(image_path)),  # Full relative path
        ]
        
        found_path = None
        for path in possible_paths:
            if os.path.exists(path):
                found_path = path
                break
        
        if found_path:
            item['image_path'] = found_path
            valid_data.append(item)
        else:
            print(f"Warning: Could not find image {image_path} (item {i+1})")
            invalid_count += 1
    
    print(f"Valid images: {len(valid_data)}")
    print(f"Invalid images: {invalid_count}")
    
    return valid_data


def add_metadata(data: List[Dict]) -> List[Dict]:
    """Add useful metadata to training examples"""
    for i, item in enumerate(data):
        # Add unique ID
        item['id'] = f"safety_example_{i+1}"
        
        # Add task category based on task_type
        task_type = item.get('task_type', 'unknown')
        if 'safety_score' in task_type:
            item['category'] = 'safety_scoring'
        elif 'binary' in task_type:
            item['category'] = 'binary_classification'
        elif 'risk' in task_type:
            item['category'] = 'risk_assessment'
        elif 'detailed' in task_type:
            item['category'] = 'detailed_analysis'
        else:
            item['category'] = 'general_safety'
        
        # Add difficulty level based on response length
        response_length = len(item.get('expected_response', ''))
        if response_length < 100:
            item['difficulty'] = 'easy'
        elif response_length < 300:
            item['difficulty'] = 'medium'
        else:
            item['difficulty'] = 'hard'
    
    return data


def analyze_data_distribution(data: List[Dict]) -> Dict[str, Any]:
    """Analyze the distribution of training data"""
    analysis = {
        'total_examples': len(data),
        'task_types': {},
        'categories': {},
        'difficulty_levels': {},
        'response_lengths': []
    }
    
    for item in data:
        # Task type distribution
        task_type = item.get('task_type', 'unknown')
        analysis['task_types'][task_type] = analysis['task_types'].get(task_type, 0) + 1
        
        # Category distribution
        category = item.get('category', 'unknown')
        analysis['categories'][category] = analysis['categories'].get(category, 0) + 1
        
        # Difficulty distribution
        difficulty = item.get('difficulty', 'unknown')
        analysis['difficulty_levels'][difficulty] = analysis['difficulty_levels'].get(difficulty, 0) + 1
        
        # Response length
        response_length = len(item.get('expected_response', ''))
        analysis['response_lengths'].append(response_length)
    
    # Calculate statistics
    response_lengths = analysis['response_lengths']
    analysis['response_length_stats'] = {
        'min': min(response_lengths) if response_lengths else 0,
        'max': max(response_lengths) if response_lengths else 0,
        'mean': sum(response_lengths) / len(response_lengths) if response_lengths else 0,
        'median': sorted(response_lengths)[len(response_lengths)//2] if response_lengths else 0
    }
    
    return analysis


def create_qwen_compatible_format(data: List[Dict]) -> List[Dict]:
    """Convert data to Qwen-compatible format"""
    qwen_data = []
    
    for item in data:
        # Create Qwen-compatible conversation format
        qwen_item = {
            'id': item.get('id', ''),
            'image_path': item['image_path'],
            'prompt': item['prompt'],
            'expected_response': item['expected_response'],
            'task_type': item.get('task_type', ''),
            'category': item.get('category', ''),
            'difficulty': item.get('difficulty', ''),
            
            # Qwen-specific format
            'conversation': [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": item['prompt']}
                    ]
                },
                {
                    "role": "assistant",
                    "content": item['expected_response']
                }
            ]
        }
        
        qwen_data.append(qwen_item)
    
    return qwen_data


def split_data_balanced(data: List[Dict], train_split: float = 0.8) -> tuple:
    """Split data while maintaining balance across categories"""
    # Group by category
    category_groups = {}
    for item in data:
        category = item.get('category', 'unknown')
        if category not in category_groups:
            category_groups[category] = []
        category_groups[category].append(item)
    
    train_data = []
    val_data = []
    
    # Split each category
    for category, items in category_groups.items():
        random.shuffle(items)
        split_idx = int(len(items) * train_split)
        train_data.extend(items[:split_idx])
        val_data.extend(items[split_idx:])
    
    # Shuffle final datasets
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    return train_data, val_data


def main():
    print("Converting VLM Safety Training Data for Qwen Fine-tuning")
    print("=" * 60)
    
    # Load original data
    input_file = "vlm_safety_training_data.json"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    with open(input_file, 'r') as f:
        original_data = json.load(f)
    
    print(f"Loaded {len(original_data)} examples from {input_file}")
    
    # Validate image paths
    valid_data = validate_image_paths(original_data)
    
    if not valid_data:
        print("Error: No valid images found!")
        return
    
    # Add metadata
    enhanced_data = add_metadata(valid_data)
    
    # Analyze data distribution
    analysis = analyze_data_distribution(enhanced_data)
    
    print("\nData Analysis:")
    print(f"Total examples: {analysis['total_examples']}")
    print(f"Task types: {analysis['task_types']}")
    print(f"Categories: {analysis['categories']}")
    print(f"Difficulty levels: {analysis['difficulty_levels']}")
    print(f"Response length stats: {analysis['response_length_stats']}")
    
    # Convert to Qwen format
    qwen_data = create_qwen_compatible_format(enhanced_data)
    
    # Split data
    train_data, val_data = split_data_balanced(qwen_data)
    
    print(f"\nData split:")
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    
    # Save processed data
    output_files = {
        'qwen_training_data.json': qwen_data,
        'qwen_train.json': train_data,
        'qwen_val.json': val_data,
        'data_analysis.json': analysis
    }
    
    for filename, data_to_save in output_files.items():
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        print(f"Saved {filename}")
    
    print("\nData conversion completed!")
    print("\nNext steps:")
    print("1. Install requirements: pip install -r qwen_requirements.txt")
    print("2. Run training: python qwen_safety_finetune.py")
    print("3. Use the generated qwen_train.json and qwen_val.json for training")


if __name__ == "__main__":
    # Set random seed for reproducible splits
    random.seed(42)
    main()
