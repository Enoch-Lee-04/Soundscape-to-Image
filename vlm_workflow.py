"""
VLM Safety Analysis Workflow
Generate training data for Vision-Language Model fine-tuning on street safety assessment
"""

import os
import json
import random
from pathlib import Path

class VLMSafetyWorkflow:
    def __init__(self, sample_svi_path="Sample SVI"):
        self.sample_svi_path = sample_svi_path
        self.image_files = self._get_image_files()
        
    def _get_image_files(self):
        """Get all image files from the Sample SVI directory"""
        if not os.path.exists(self.sample_svi_path):
            print(f"Warning: {self.sample_svi_path} directory not found")
            return []
        
        image_files = []
        for file in os.listdir(self.sample_svi_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(self.sample_svi_path, file))
        
        return sorted(image_files)
    
    def generate_safety_score_prompt(self, image_path):
        """Generate a safety scoring prompt"""
        return f"""Analyze this street view image and provide a safety score from 1-10 (10 being safest).

Consider these factors:
- Sidewalk condition and width
- Traffic density and speed  
- Street lighting quality
- Road maintenance
- Pedestrian-vehicle separation
- Visibility and sightlines

Provide your response in this format:
Safety Score: [1-10]
Reasoning: [Brief explanation of your assessment]

Image: {image_path}"""

    def generate_binary_classification_prompt(self, image_path):
        """Generate a binary safety classification prompt"""
        return f"""Classify this street view image as SAFE or UNSAFE for pedestrians.

SAFE = Good sidewalks, low traffic, good lighting, well-maintained infrastructure
UNSAFE = Poor sidewalks, heavy traffic, poor lighting, infrastructure issues

Provide your response as:
Classification: [SAFE/UNSAFE]
Confidence: [HIGH/MEDIUM/LOW]
Reason: [Brief explanation]

Image: {image_path}"""

    def generate_detailed_analysis_prompt(self, image_path):
        """Generate a detailed safety analysis prompt"""
        return f"""Provide a detailed safety analysis of this street view image.

Evaluate these aspects (1-10 scale each):
1. Pedestrian Safety: ___
2. Traffic Safety: ___
3. Lighting Safety: ___
4. Infrastructure Safety: ___
5. Crime Safety: ___

Overall Score: ___/50
Main Concerns: [List top 3 safety issues]
Strengths: [List top 3 positive safety features]

Image: {image_path}"""

    def generate_risk_assessment_prompt(self, image_path):
        """Generate a risk assessment prompt"""
        return f"""Assess the safety risks in this street view image.

Identify risks by category:
HIGH RISK: [List high-risk elements]
MEDIUM RISK: [List medium-risk elements]  
LOW RISK: [List low-risk elements]

Overall Risk Level: [LOW/MEDIUM/HIGH]
Primary Risk Factor: [Main safety concern]

Image: {image_path}"""

    def create_training_dataset(self, num_samples_per_image=3):
        """Create a training dataset with multiple prompt types per image"""
        
        if not self.image_files:
            print("No image files found!")
            return []
        
        training_data = []
        prompt_generators = [
            ("safety_score", self.generate_safety_score_prompt),
            ("binary_classification", self.generate_binary_classification_prompt),
            ("detailed_analysis", self.generate_detailed_analysis_prompt),
            ("risk_assessment", self.generate_risk_assessment_prompt)
        ]

        for image_path in self.image_files:
            # Randomly select prompt types for each image
            selected_prompts = random.sample(prompt_generators, 
                                        min(num_samples_per_image, len(prompt_generators)))

            for task_type, prompt_generator in selected_prompts:
                training_data.append({
                    "image_path": image_path,
                    "prompt": prompt_generator(image_path),
                    "task_type": task_type,
                    "expected_format": self._get_expected_format(task_type)
                })

        return training_data

    def _get_expected_format(self, task_type):
        """Get the expected response format for each task type"""
        formats = {
            "safety_score": "Safety Score: [1-10]\nReasoning: [explanation]",
            "binary_classification": "Classification: [SAFE/UNSAFE]\nConfidence: [HIGH/MEDIUM/LOW]\nReason: [explanation]",
            "detailed_analysis": "Detailed scores for 5 safety aspects + Overall Score + Concerns + Strengths",
            "risk_assessment": "Risk categories + Overall Risk Level + Primary Risk Factor"
        }
        return formats.get(task_type, "Structured analysis")

    def save_training_data(self, training_data, filename="vlm_safety_training_data.json"):
        """Save training data to JSON file"""
        with open(filename, 'w') as f:
            json.dump(training_data, f, indent=2)

        print(f"Training data saved to {filename}")
        print(f"Total training examples: {len(training_data)}")
        print(f"Images processed: {len(self.image_files)}")

    def print_sample_prompts(self, num_samples=3):
        """Print sample prompts for review"""
        if not self.image_files:
            print("No image files found!")
            return

        print("=== SAMPLE VLM SAFETY PROMPTS ===\n")

        for i, image_path in enumerate(self.image_files[:num_samples]):
            print(f"SAMPLE {i+1}: {image_path}")
            print("-" * 50)
            
            # Show different prompt types
            print("1. SAFETY SCORE PROMPT:")
            print(self.generate_safety_score_prompt(image_path))
            print("\n2. BINARY CLASSIFICATION PROMPT:")
            print(self.generate_binary_classification_prompt(image_path))
            print("\n" + "="*80 + "\n")

def main():
    """Main workflow function"""
    print("VLM Safety Analysis Workflow")
    print("=" * 40)

    # Initialize workflow
    workflow = VLMSafetyWorkflow()

    if not workflow.image_files:
        print("No image files found in Sample SVI directory!")
        return

    print(f"Found {len(workflow.image_files)} images in Sample SVI directory")

    # Show sample prompts
    workflow.print_sample_prompts(num_samples=2)

    # Generate training dataset
    print("Generating training dataset...")
    training_data = workflow.create_training_dataset(num_samples_per_image=3)

    # Save training data
    workflow.save_training_data(training_data)

    print("\n=== NEXT STEPS ===")
    print("1. Review the generated prompts in vlm_safety_training_data.json")
    print("2. Manually annotate responses for each prompt (or use GPT-4 for initial annotations)")
    print("3. Use the annotated data to fine-tune your VLM")
    print("4. Test the fine-tuned model on new street view images")

if __name__ == "__main__":
    main()
