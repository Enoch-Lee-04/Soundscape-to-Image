"""
Simple Image Path Fixer
Fixes image paths without requiring PIL
"""

import os
import json

def fix_image_paths():
    """Fix image paths in the JSON file"""
    
    print("Fixing image paths...")
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
            print(f"✓ Fixed: {original_path} -> {fixed_path}")
        else:
            print(f"✗ Could not fix: {original_path}")
    
    # Save the fixed data
    with open('vlm_safety_training_data.json', 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"\nFixed {fixed_count} image paths!")
    print("Updated vlm_safety_training_data.json")

def test_paths():
    """Test if paths work"""
    print("Testing image paths...")
    print("=" * 50)
    
    # Load the training data
    with open('vlm_safety_training_data.json', 'r') as f:
        data = json.load(f)
    
    working_paths = 0
    
    for i, entry in enumerate(data[:5]):  # Test first 5
        image_path = entry['image_path']
        if os.path.exists(image_path):
            print(f"✓ Entry {i+1}: {image_path}")
            working_paths += 1
        else:
            print(f"✗ Entry {i+1}: {image_path}")
    
    print(f"\nWorking paths: {working_paths}/5")

if __name__ == "__main__":
    print("VLM Safety Image Path Fixer")
    print("=" * 40)
    
    # Test current paths
    test_paths()
    
    # Fix paths
    fix_image_paths()
    
    # Test again
    print("\nTesting after fix...")
    test_paths()
    
    print("\nDone! You can now run the annotation helper.")
