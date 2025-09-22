"""
Setup Script for Qwen VLM Safety Assessment Training
Prepares environment and data for fine-tuning
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required!")
        sys.exit(1)
    print(f"✓ Python version: {sys.version}")


def check_cuda_availability():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠ CUDA not available, will use CPU")
            return False
    except ImportError:
        print("⚠ PyTorch not installed yet")
        return False


def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "qwen_requirements.txt"
        ])
        print("✓ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False


def check_data_files():
    """Check if required data files exist"""
    required_files = [
        "vlm_safety_training_data.json",
        "Sample SVI"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"✗ Missing required files: {missing_files}")
        return False
    
    print("✓ All required data files found")
    
    # Check if Sample SVI folder has images
    svi_folder = Path("Sample SVI")
    image_files = list(svi_folder.glob("*.jpg")) + list(svi_folder.glob("*.jpeg")) + list(svi_folder.glob("*.png"))
    
    if not image_files:
        print("✗ No images found in Sample SVI folder!")
        return False
    
    print(f"✓ Found {len(image_files)} images in Sample SVI folder")
    return True


def convert_data():
    """Convert data to Qwen format"""
    print("Converting data to Qwen format...")
    try:
        subprocess.check_call([sys.executable, "convert_data_for_qwen.py"])
        print("✓ Data conversion completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error converting data: {e}")
        return False


def create_training_config():
    """Create training configuration file"""
    config = {
        "model_args": {
            "model_name_or_path": "Qwen/Qwen2-VL-2B-Instruct",
            "trust_remote_code": True,
            "use_cache": False
        },
        "data_args": {
            "data_path": "qwen_training_data.json",
            "image_folder": "Sample SVI",
            "max_length": 2048,
            "train_split": 0.8
        },
        "training_args": {
            "output_dir": "./qwen_safety_model",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 5e-5,
            "warmup_ratio": 0.1,
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500,
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "report_to": "wandb",
            "run_name": "qwen_safety_assessment"
        }
    }
    
    with open("training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✓ Training configuration created: training_config.json")


def create_training_script():
    """Create a simple training script"""
    script_content = '''#!/usr/bin/env python3
"""
Simple training script for Qwen VLM Safety Assessment
"""

import subprocess
import sys
import os

def main():
    print("Starting Qwen VLM Safety Assessment Training")
    print("=" * 50)
    
    # Check if model exists
    if os.path.exists("./qwen_safety_model"):
        print("Found existing model directory. Continuing training...")
    
    # Run training
    cmd = [
        sys.executable, "qwen_safety_finetune.py",
        "--model_name_or_path", "Qwen/Qwen2-VL-2B-Instruct",
        "--data_path", "qwen_training_data.json",
        "--output_dir", "./qwen_safety_model",
        "--num_train_epochs", "3",
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "4",
        "--learning_rate", "5e-5",
        "--logging_steps", "10",
        "--save_steps", "500",
        "--eval_steps", "500",
        "--report_to", "wandb",
        "--run_name", "qwen_safety_assessment"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\\nTraining completed successfully!")
        print("Model saved to: ./qwen_safety_model")
        
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
    
    with open("train_qwen.py", "w") as f:
        f.write(script_content)
    
    # Make it executable on Unix systems
    try:
        os.chmod("train_qwen.py", 0o755)
    except:
        pass
    
    print("✓ Training script created: train_qwen.py")


def create_inference_script():
    """Create a simple inference script"""
    script_content = '''#!/usr/bin/env python3
"""
Simple inference script for Qwen VLM Safety Assessment
"""

import subprocess
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_inference.py <image_path> [prompt_type]")
        print("Prompt types: safety_score, binary_classification, detailed_analysis, risk_assessment")
        return 1
    
    image_path = sys.argv[1]
    prompt_type = sys.argv[2] if len(sys.argv) > 2 else "safety_score"
    
    if not os.path.exists("./qwen_safety_model"):
        print("Error: Trained model not found at ./qwen_safety_model")
        print("Please run training first: python train_qwen.py")
        return 1
    
    cmd = [
        sys.executable, "qwen_safety_inference.py",
        "--model_path", "./qwen_safety_model",
        "--image_path", image_path,
        "--prompt_type", prompt_type,
        "--output_file", f"safety_assessment_{os.path.basename(image_path)}.json"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\\nInference completed for {image_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"Inference failed with error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
    
    with open("run_inference.py", "w") as f:
        f.write(script_content)
    
    try:
        os.chmod("run_inference.py", 0o755)
    except:
        pass
    
    print("✓ Inference script created: run_inference.py")


def main():
    print("Qwen VLM Safety Assessment Training Setup")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Check CUDA
    cuda_available = check_cuda_availability()
    
    # Check data files
    if not check_data_files():
        print("Please ensure all required data files are present before continuing.")
        return 1
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Please install manually:")
        print("pip install -r qwen_requirements.txt")
        return 1
    
    # Convert data
    if not convert_data():
        print("Failed to convert data. Please run convert_data_for_qwen.py manually.")
        return 1
    
    # Create configuration and scripts
    create_training_config()
    create_training_script()
    create_inference_script()
    
    print("\\n" + "=" * 50)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("\\nNext steps:")
    print("1. Review training_config.json and adjust parameters if needed")
    print("2. Start training: python train_qwen.py")
    print("3. Monitor training with wandb (if configured)")
    print("4. Run inference: python run_inference.py <image_path>")
    
    if not cuda_available:
        print("\\n⚠ Note: CUDA not available. Training will be slower on CPU.")
        print("Consider using Google Colab or a GPU-enabled environment for faster training.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
