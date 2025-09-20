"""
Simple VLM Safety Analysis Prompts
Ready-to-use prompts for street view safety assessment
"""

# PROMPT TEMPLATES FOR VLM FINE-TUNING

# 1. OVERALL SAFETY SCORING
SAFETY_SCORE_PROMPT = """Analyze this street view image and provide a safety score from 1-10 (10 being safest).

Consider:
- Sidewalk condition and width
- Traffic density and speed
- Street lighting
- Road maintenance
- Pedestrian-vehicle separation
- Visibility and sightlines

Format your response as:
Safety Score: [1-10]
Reasoning: [Brief explanation]"""

# 2. BINARY SAFETY CLASSIFICATION
BINARY_SAFETY_PROMPT = """Classify this street view image as SAFE or UNSAFE for pedestrians.

SAFE = Good sidewalks, low traffic, good lighting, well-maintained infrastructure
UNSAFE = Poor sidewalks, heavy traffic, poor lighting, infrastructure issues

Response format:
Classification: [SAFE/UNSAFE]
Confidence: [HIGH/MEDIUM/LOW]
Reason: [Brief explanation]"""

# 3. DETAILED SAFETY ANALYSIS
DETAILED_SAFETY_PROMPT = """Provide a detailed safety analysis of this street view image.

Evaluate these aspects (1-10 scale each):
1. Pedestrian Safety: ___
2. Traffic Safety: ___
3. Lighting Safety: ___
4. Infrastructure Safety: ___
5. Crime Safety: ___

Overall Score: ___/50
Main Concerns: [List top 3 issues]
Strengths: [List top 3 positive features]"""

# 4. COMPARATIVE ANALYSIS
COMPARATIVE_PROMPT = """Compare this street view against safety best practices.

Rate each dimension (1-10):
- Sidewalk Quality: ___
- Traffic Management: ___
- Lighting Coverage: ___
- Infrastructure Maintenance: ___
- Pedestrian Comfort: ___

Overall Rating: ___/50
Recommendations: [Top 3 improvements needed]"""

# 5. RISK ASSESSMENT
RISK_ASSESSMENT_PROMPT = """Assess the safety risks in this street view image.

Identify risks by category:
HIGH RISK: [List high-risk elements]
MEDIUM RISK: [List medium-risk elements]
LOW RISK: [List low-risk elements]

Overall Risk Level: [LOW/MEDIUM/HIGH]
Primary Risk Factor: [Main safety concern]"""

# 6. ACCESSIBILITY FOCUS
ACCESSIBILITY_PROMPT = """Evaluate this street view for accessibility and safety.

Rate accessibility features (1-10):
- Wheelchair Access: ___
- Visual Accessibility: ___
- Mobility Device Support: ___
- Clear Pathways: ___

Accessibility Score: ___/40
Barriers Identified: [List accessibility issues]
Improvements Needed: [Suggestions for better accessibility]"""

# EXAMPLE TRAINING DATA FORMAT
EXAMPLE_TRAINING_DATA = [
    {
        "image_path": "Sample SVI/1.jpg",
        "prompt": SAFETY_SCORE_PROMPT,
        "expected_response": "Safety Score: 7\nReasoning: Good sidewalk width and condition, moderate traffic, adequate lighting, but some areas need better maintenance.",
        "task_type": "safety_scoring"
    },
    {
        "image_path": "Sample SVI/2.jpg", 
        "prompt": BINARY_SAFETY_PROMPT,
        "expected_response": "Classification: SAFE\nConfidence: HIGH\nReason: Wide sidewalks, low traffic, good lighting, well-maintained infrastructure.",
        "task_type": "binary_classification"
    }
]

# QUICK PROMPT GENERATOR FUNCTION
def generate_prompt_for_image(image_path: str, prompt_type: str = "safety_score") -> str:
    """Generate a prompt for a specific image and task type"""
    
    prompt_templates = {
        "safety_score": SAFETY_SCORE_PROMPT,
        "binary": BINARY_SAFETY_PROMPT,
        "detailed": DETAILED_SAFETY_PROMPT,
        "comparative": COMPARATIVE_PROMPT,
        "risk": RISK_ASSESSMENT_PROMPT,
        "accessibility": ACCESSIBILITY_PROMPT
    }
    
    base_prompt = prompt_templates.get(prompt_type, SAFETY_SCORE_PROMPT)
    return f"Image: {image_path}\n\n{base_prompt}"

# USAGE EXAMPLES
if __name__ == "__main__":
    # Example: Generate prompts for all images
    image_paths = [f"Sample SVI/{i}.jpg" for i in range(1, 31)]
    
    print("=== SAMPLE PROMPTS FOR VLM FINE-TUNING ===\n")
    
    # Generate different types of prompts for the first image
    sample_image = "Sample SVI/1.jpg"
    
    print("1. SAFETY SCORE PROMPT:")
    print(generate_prompt_for_image(sample_image, "safety_score"))
    print("\n" + "="*50 + "\n")
    
    print("2. BINARY CLASSIFICATION PROMPT:")
    print(generate_prompt_for_image(sample_image, "binary"))
    print("\n" + "="*50 + "\n")
    
    print("3. DETAILED ANALYSIS PROMPT:")
    print(generate_prompt_for_image(sample_image, "detailed"))
    print("\n" + "="*50 + "\n")
    
    print("4. RISK ASSESSMENT PROMPT:")
    print(generate_prompt_for_image(sample_image, "risk"))
    print("\n" + "="*50 + "\n")
    
    print("5. ACCESSIBILITY PROMPT:")
    print(generate_prompt_for_image(sample_image, "accessibility"))
