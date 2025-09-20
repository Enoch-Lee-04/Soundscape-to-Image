"""
Quick Fix for Image Path Issues
Run this to test image loading and fix path issues
"""

import os
import json
from PIL import Image

def test_image_loading():
    """Test loading images with different path formats"""
    
    # Load the training data
    with open('vlm_safety_training_data.json', 'r') as f:
        data = json.load(f)
    
    print("Testing image loading...")
    print("=" * 50)
    
    # Test first few images
    for i in range(min(5, len(data))):
        entry = data[i]
        image_path = entry['image_path']
        
        print(f"\nEntry {i+1}: {image_path}")
        
        # Test different path formats
        paths_to_test = [
            image_path,  # Original path
            image_path.replace('\\', '/'),  # Forward slashes
            os.path.join(os.getcwd(), image_path),  # Absolute path
            os.path.normpath(os.path.join(os.getcwd(), image_path))  # Normalized path
        ]
        
        for j, test_path in enumerate(paths_to_test):
            try:
                if os.path.exists(test_path):
                    # Try to open with PIL
                    img = Image.open(test_path)
                    print(f"  ✓ Path {j+1} works: {test_path}")
                    print(f"    Image size: {img.size}")
                    break
                else:
                    print(f"  ✗ Path {j+1} not found: {test_path}")
            except Exception as e:
                print(f"  ✗ Path {j+1} error: {e}")
        else:
            print("  ✗ No working path found!")

def fix_image_paths():
    """Fix image paths in the JSON file"""
    
    print("\nFixing image paths...")
    print("=" * 50)
    
    # Load the training data
    with open('vlm_safety_training_data.json', 'r') as f:
        data = json.load(f)
    
    fixed_count = 0
    
    for entry in data:
        original_path = entry['image_path']
        
        # Convert to proper Windows path
        fixed_path = os.path.normpath(os.path.join(os.getcwd(), original_path))
        
        if os.path.exists(fixed_path):
            entry['image_path'] = fixed_path
            fixed_count += 1
        else:
            print(f"Warning: Could not fix path: {original_path}")
    
    # Save the fixed data
    with open('vlm_safety_training_data.json', 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Fixed {fixed_count} image paths!")
    print("Updated vlm_safety_training_data.json")

if __name__ == "__main__":
    print("VLM Safety Image Path Fixer")
    print("=" * 40)
    
    # Test current paths
    test_image_loading()
    
    # Ask if user wants to fix paths
    print("\n" + "=" * 50)
    choice = input("Do you want to fix the image paths? (y/n): ").lower()
    
    if choice == 'y':
        fix_image_paths()
        print("\nPaths fixed! You can now run the annotation helper.")
    else:
        print("Paths not changed. You may need to fix them manually.")
