"""
Install Qwen requirements in the correct order
Handles the PyTorch -> DeepSpeed dependency issue
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"Installing {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ {description} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    print("Installing Qwen VLM requirements in correct order...")
    print("=" * 60)
    
    # Step 1: Install PyTorch first
    pytorch_cmd = [
        sys.executable, "-m", "pip", "install", 
        "torch>=2.0.0", "torchvision>=0.15.0", "torchaudio>=2.0.0"
    ]
    
    if not run_command(pytorch_cmd, "PyTorch packages"):
        print("Failed to install PyTorch. Please install manually:")
        print("pip install torch torchvision torchaudio")
        return 1
    
    # Step 2: Install core packages
    core_packages = [
        "transformers>=4.37.0",
        "datasets>=2.14.0", 
        "accelerate>=0.20.0",
        "peft>=0.6.0",
        "qwen-agent>=0.0.12"
    ]
    
    for package in core_packages:
        cmd = [sys.executable, "-m", "pip", "install", package]
        if not run_command(cmd, package):
            print(f"Warning: Failed to install {package}")
    
    # Step 3: Install image processing packages
    image_packages = [
        "Pillow>=9.0.0",
        "opencv-python>=4.5.0", 
        "matplotlib>=3.5.0"
    ]
    
    for package in image_packages:
        cmd = [sys.executable, "-m", "pip", "install", package]
        if not run_command(cmd, package):
            print(f"Warning: Failed to install {package}")
    
    # Step 4: Install training utilities
    training_packages = [
        "wandb>=0.15.0",
        "tensorboard>=2.10.0",
        "tqdm>=4.64.0"
    ]
    
    for package in training_packages:
        cmd = [sys.executable, "-m", "pip", "install", package]
        if not run_command(cmd, package):
            print(f"Warning: Failed to install {package}")
    
    # Step 5: Install evaluation packages
    eval_packages = [
        "scikit-learn>=1.1.0",
        "numpy>=1.21.0", 
        "pandas>=1.4.0",
        "jsonlines>=3.0.0"
    ]
    
    for package in eval_packages:
        cmd = [sys.executable, "-m", "pip", "install", package]
        if not run_command(cmd, package):
            print(f"Warning: Failed to install {package}")
    
    # Step 6: Try to install DeepSpeed (optional)
    print("Attempting to install DeepSpeed (optional)...")
    deepspeed_cmd = [sys.executable, "-m", "pip", "install", "deepspeed>=0.10.0"]
    if run_command(deepspeed_cmd, "DeepSpeed"):
        print("✓ DeepSpeed installed successfully")
    else:
        print("⚠ DeepSpeed installation failed (this is optional for training)")
        print("You can continue without DeepSpeed - it's only needed for advanced distributed training")
    
    print("\n" + "=" * 60)
    print("INSTALLATION COMPLETED!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Verify installation: python -c 'import torch; print(torch.__version__)'")
    print("2. Start training: python qwen_safety_finetune.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
