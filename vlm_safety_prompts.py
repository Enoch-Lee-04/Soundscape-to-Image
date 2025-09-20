"""
VLM Safety Analysis Prompts for Street View Images
Designed for fine-tuning Vision-Language Models on safety assessment tasks
"""

import json
import random
from typing import List, Dict, Any

class SafetyPromptGenerator:
    def __init__(self):
        self.safety_aspects = [
            "pedestrian_safety",
            "traffic_safety", 
            "lighting_safety",
            "infrastructure_safety",
            "crime_safety",
            "overall_safety"
        ]
        
        self.safety_indicators = {
            "pedestrian_safety": [
                "sidewalk condition", "crosswalk visibility", "pedestrian traffic", 
                "barriers between pedestrians and vehicles", "walking space"
            ],
            "traffic_safety": [
                "traffic density", "speed limit compliance", "road condition",
                "traffic signs visibility", "vehicle behavior"
            ],
            "lighting_safety": [
                "street lighting", "visibility at night", "shadow areas",
                "lighting uniformity", "dark spots"
            ],
            "infrastructure_safety": [
                "road maintenance", "sidewalk condition", "drainage",
                "signage clarity", "barrier condition"
            ],
            "crime_safety": [
                "visibility", "escape routes", "crowding", 
                "surveillance presence", "maintenance level"
            ]
        }

    def generate_safety_score_prompt(self, image_path: str, aspect: str = "overall") -> Dict[str, Any]:
        """Generate a prompt for safety scoring"""
        
        if aspect == "overall":
            prompt = f"""Analyze this street view image and provide a comprehensive safety assessment.

Please evaluate the following aspects:
1. Pedestrian Safety: Sidewalk condition, crosswalk visibility, pedestrian-vehicle separation
2. Traffic Safety: Traffic density, road condition, signage visibility
3. Lighting Safety: Street lighting quality, visibility, shadow areas
4. Infrastructure Safety: Road maintenance, sidewalk condition, drainage
5. Crime Safety: Visibility, escape routes, surveillance presence

Provide your assessment in this format:
- Overall Safety Score: [1-10 scale, where 10 is safest]
- Detailed Analysis: [Brief explanation of key safety factors]
- Primary Concerns: [List main safety issues if any]
- Positive Features: [List safety strengths]

Image: {image_path}"""

        else:
            indicators = self.safety_indicators.get(aspect, [])
            prompt = f"""Analyze this street view image specifically for {aspect.replace('_', ' ')} safety.

Focus on these indicators: {', '.join(indicators)}

Provide your assessment in this format:
- {aspect.replace('_', ' ').title()} Safety Score: [1-10 scale]
- Key Observations: [Specific details about {aspect.replace('_', ' ')}]
- Recommendations: [Suggestions for improvement]

Image: {image_path}"""

        return {
            "image_path": image_path,
            "prompt": prompt,
            "task_type": "safety_scoring",
            "aspect": aspect,
            "expected_output_format": "structured_analysis"
        }

    def generate_comparative_prompt(self, image_path: str) -> Dict[str, Any]:
        """Generate a prompt for comparative safety analysis"""
        
        prompt = f"""Compare this street view image against safety best practices.

Rate each safety dimension (1-10 scale) and explain your reasoning:

1. PEDESTRIAN SAFETY (1-10):
   - Sidewalk width and condition: ___
   - Crosswalk visibility: ___
   - Separation from traffic: ___
   - Reasoning: [Your analysis]

2. TRAFFIC SAFETY (1-10):
   - Traffic flow and density: ___
   - Road condition: ___
   - Signage clarity: ___
   - Reasoning: [Your analysis]

3. LIGHTING SAFETY (1-10):
   - Street lighting quality: ___
   - Visibility at night: ___
   - Shadow areas: ___
   - Reasoning: [Your analysis]

4. INFRASTRUCTURE SAFETY (1-10):
   - Road maintenance: ___
   - Drainage system: ___
   - Barrier condition: ___
   - Reasoning: [Your analysis]

5. CRIME SAFETY (1-10):
   - Visibility and sightlines: ___
   - Escape routes: ___
   - Surveillance presence: ___
   - Reasoning: [Your analysis]

OVERALL SAFETY SCORE: ___/50
SUMMARY: [Brief overall assessment]

Image: {image_path}"""

        return {
            "image_path": image_path,
            "prompt": prompt,
            "task_type": "comparative_analysis",
            "expected_output_format": "detailed_scoring"
        }

    def generate_binary_classification_prompt(self, image_path: str) -> Dict[str, Any]:
        """Generate a prompt for binary safety classification"""
        
        prompt = f"""Classify this street view image as either "SAFE" or "UNSAFE" for pedestrians.

Consider these factors:
- Sidewalk condition and width
- Traffic density and speed
- Lighting quality
- Infrastructure maintenance
- Visibility and sightlines
- Pedestrian-vehicle separation

Your response should be:
SAFETY CLASSIFICATION: [SAFE/UNSAFE]
CONFIDENCE LEVEL: [HIGH/MEDIUM/LOW]
KEY REASONING: [Brief explanation of your decision]

Image: {image_path}"""

        return {
            "image_path": image_path,
            "prompt": prompt,
            "task_type": "binary_classification",
            "expected_output_format": "classification_with_reasoning"
        }

    def generate_detailed_analysis_prompt(self, image_path: str) -> Dict[str, Any]:
        """Generate a prompt for detailed safety analysis"""
        
        prompt = f"""Provide a comprehensive safety analysis of this street view image.

SAFETY ASSESSMENT FRAMEWORK:

1. IMMEDIATE HAZARDS:
   - List any immediate safety threats
   - Rate severity (Low/Medium/High)

2. INFRASTRUCTURE ANALYSIS:
   - Sidewalk condition and width
   - Road surface quality
   - Drainage system
   - Street furniture condition

3. TRAFFIC ASSESSMENT:
   - Vehicle speed and density
   - Traffic control measures
   - Signage visibility
   - Pedestrian-vehicle interaction

4. LIGHTING EVALUATION:
   - Street lighting coverage
   - Shadow areas
   - Night visibility
   - Lighting uniformity

5. CRIME PREVENTION FEATURES:
   - Natural surveillance opportunities
   - Escape routes
   - Maintenance level
   - Activity level

6. ACCESSIBILITY CONSIDERATIONS:
   - Wheelchair accessibility
   - Visual impairment considerations
   - Mobility device accommodation

7. RECOMMENDATIONS:
   - Priority improvements
   - Cost-effective solutions
   - Long-term planning suggestions

Image: {image_path}"""

        return {
            "image_path": image_path,
            "prompt": prompt,
            "task_type": "detailed_analysis",
            "expected_output_format": "comprehensive_report"
        }

    def generate_training_dataset(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Generate a complete training dataset with various prompt types"""

        dataset = []

        for image_path in image_paths:
            # Generate different types of prompts for each image
            dataset.append(self.generate_safety_score_prompt(image_path, "overall"))
            dataset.append(self.generate_safety_score_prompt(image_path, "pedestrian_safety"))
            dataset.append(self.generate_safety_score_prompt(image_path, "traffic_safety"))
            dataset.append(self.generate_comparative_prompt(image_path))
            dataset.append(self.generate_binary_classification_prompt(image_path))
            dataset.append(self.generate_detailed_analysis_prompt(image_path))

        return dataset

    def save_training_data(self, dataset: List[Dict[str, Any]], filename: str = "vlm_safety_training_data.json"):
        """Save the training dataset to a JSON file"""

        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)

        print(f"Training dataset saved to {filename}")
        print(f"Total training examples: {len(dataset)}")

def main():
    """Example usage of the SafetyPromptGenerator"""

    # Initialize the generator
    generator = SafetyPromptGenerator()

    # Sample image paths (you can replace these with actual paths)
    sample_images = [f"Sample SVI/{i}.jpg" for i in range(1, 31)]

    # Generate training dataset
    training_data = generator.generate_training_dataset(sample_images)

    # Save the dataset
    generator.save_training_data(training_data)

    # Print a sample prompt
    print("\n" + "="*50)
    print("SAMPLE PROMPT:")
    print("="*50)
    sample_prompt = generator.generate_safety_score_prompt("Sample SVI/1.jpg", "overall")
    print(sample_prompt["prompt"])

if __name__ == "__main__":
    main()
