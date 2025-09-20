"""
Simple Annotation Helper (No PIL Required)
Opens images in default viewer and provides command-line interface
"""

import json
import os
import subprocess
import sys

class SimpleAnnotationHelper:
    def __init__(self, data_file="vlm_safety_training_data.json"):
        self.data_file = data_file
        self.training_data = []
        self.current_index = 0
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load the training data from JSON file"""
        try:
            with open(self.data_file, 'r') as f:
                self.training_data = json.load(f)
            print(f"Loaded {len(self.training_data)} training examples")
        except FileNotFoundError:
            print(f"Error: Could not find {self.data_file}")
            sys.exit(1)
    
    def open_image(self, image_path):
        """Open image using default system viewer"""
        try:
            if os.path.exists(image_path):
                if sys.platform.startswith('win'):
                    os.startfile(image_path)
                elif sys.platform.startswith('darwin'):  # macOS
                    subprocess.run(['open', image_path])
                else:  # Linux
                    subprocess.run(['xdg-open', image_path])
                return True
            else:
                print(f"Image not found: {image_path}")
                return False
        except Exception as e:
            print(f"Error opening image: {e}")
            return False
    
    def annotate_entry(self, entry):
        """Annotate a single entry"""
        print("\n" + "="*60)
        print(f"Entry {self.current_index + 1} of {len(self.training_data)}")
        print(f"Image: {entry['image_path']}")
        print(f"Task Type: {entry['task_type'].replace('_', ' ').title()}")
        print("="*60)
        
        # Open image
        print("Opening image...")
        self.open_image(entry['image_path'])
        
        # Show prompt
        print("\nPrompt:")
        print("-" * 40)
        print(entry['prompt'])
        print("-" * 40)
        
        # Get annotation based on task type
        task_type = entry['task_type']
        
        if task_type == 'safety_score':
            response = self.annotate_safety_score()
        elif task_type == 'binary_classification':
            response = self.annotate_binary_classification()
        elif task_type == 'detailed_analysis':
            response = self.annotate_detailed_analysis()
        elif task_type == 'risk_assessment':
            response = self.annotate_risk_assessment()
        else:
            print(f"Unknown task type: {task_type}")
            response = input("Enter your response: ")
        
        return response
    
    def annotate_safety_score(self):
        """Annotate safety score task"""
        print("\nSafety Score Annotation:")
        
        while True:
            try:
                score = int(input("Enter safety score (1-10): "))
                if 1 <= score <= 10:
                    break
                else:
                    print("Score must be between 1 and 10")
            except ValueError:
                print("Please enter a valid number")
        
        reasoning = input("Enter reasoning: ")
        
        return f"Safety Score: {score}\nReasoning: {reasoning}"
    
    def annotate_binary_classification(self):
        """Annotate binary classification task"""
        print("\nBinary Classification Annotation:")
        
        while True:
            classification = input("Enter classification (SAFE/UNSAFE): ").upper()
            if classification in ['SAFE', 'UNSAFE']:
                break
            else:
                print("Please enter SAFE or UNSAFE")
        
        while True:
            confidence = input("Enter confidence (HIGH/MEDIUM/LOW): ").upper()
            if confidence in ['HIGH', 'MEDIUM', 'LOW']:
                break
            else:
                print("Please enter HIGH, MEDIUM, or LOW")
        
        reason = input("Enter reason: ")
        
        return f"Classification: {classification}\nConfidence: {confidence}\nReason: {reason}"
    
    def annotate_detailed_analysis(self):
        """Annotate detailed analysis task"""
        print("\nDetailed Analysis Annotation:")
        
        aspects = ["Pedestrian Safety", "Traffic Safety", "Lighting Safety", "Infrastructure Safety", "Crime Safety"]
        scores = []
        total = 0
        
        for aspect in aspects:
            while True:
                try:
                    score = int(input(f"Enter {aspect} score (1-10): "))
                    if 1 <= score <= 10:
                        scores.append(f"{aspect}: {score}")
                        total += score
                        break
                    else:
                        print("Score must be between 1 and 10")
                except ValueError:
                    print("Please enter a valid number")
        
        concerns = input("Enter main concerns: ")
        strengths = input("Enter strengths: ")
        
        response = f"{scores[0]}\n{scores[1]}\n{scores[2]}\n{scores[3]}\n{scores[4]}\n\nOverall Score: {total}/50\nMain Concerns: {concerns}\nStrengths: {strengths}"
        
        return response
    
    def annotate_risk_assessment(self):
        """Annotate risk assessment task"""
        print("\nRisk Assessment Annotation:")
        
        high_risk = input("Enter HIGH RISK elements: ")
        medium_risk = input("Enter MEDIUM RISK elements: ")
        low_risk = input("Enter LOW RISK elements: ")
        
        while True:
            overall_risk = input("Enter overall risk level (LOW/MEDIUM/HIGH): ").upper()
            if overall_risk in ['LOW', 'MEDIUM', 'HIGH']:
                break
            else:
                print("Please enter LOW, MEDIUM, or HIGH")
        
        primary_risk = input("Enter primary risk factor: ")
        
        return f"HIGH RISK: {high_risk}\nMEDIUM RISK: {medium_risk}\nLOW RISK: {low_risk}\n\nOverall Risk Level: {overall_risk}\nPrimary Risk Factor: {primary_risk}"
    
    def run(self):
        """Run the annotation process"""
        print("VLM Safety Annotation Helper (Simple Version)")
        print("=" * 50)
        
        while self.current_index < len(self.training_data):
            entry = self.training_data[self.current_index]
            
            # Check if already annotated
            if 'expected_response' in entry:
                print(f"\nEntry {self.current_index + 1} already annotated. Skipping...")
                self.current_index += 1
                continue
            
            # Annotate entry
            response = self.annotate_entry(entry)
            
            # Save annotation
            entry['expected_response'] = response
            
            # Ask for next action
            print("\nOptions:")
            print("1. Continue to next entry")
            print("2. Go back to previous entry")
            print("3. Save and exit")
            print("4. Exit without saving")
            
            while True:
                choice = input("Enter your choice (1-4): ")
                if choice == '1':
                    self.current_index += 1
                    break
                elif choice == '2':
                    if self.current_index > 0:
                        self.current_index -= 1
                    break
                elif choice == '3':
                    self.save_data()
                    print("Annotations saved successfully!")
                    return
                elif choice == '4':
                    print("Exiting without saving...")
                    return
                else:
                    print("Please enter 1, 2, 3, or 4")
        
        # Save when done
        self.save_data()
        print("\nAll entries annotated! Data saved successfully.")
    
    def save_data(self):
        """Save the annotated data"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.training_data, f, indent=4)
            print("Data saved successfully!")
        except Exception as e:
            print(f"Error saving data: {e}")

def main():
    helper = SimpleAnnotationHelper()
    helper.run()

if __name__ == "__main__":
    main()
