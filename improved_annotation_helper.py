"""
Improved VLM Safety Annotation Helper
Fixed GUI with proper layout and functionality
"""

import json
import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

class ImprovedAnnotationHelper:
    def __init__(self, root):
        self.root = root
        self.root.title("VLM Safety Annotation Helper")
        self.root.geometry("1400x900")
        
        # Data
        self.training_data = []
        self.current_index = 0
        self.data_file = "vlm_safety_training_data.json"
        
        # Load data
        self.load_data()
        
        # Create GUI
        self.create_widgets()
        
        # Display first entry
        self.display_current_entry()
    
    def load_data(self):
        """Load the training data from JSON file"""
        try:
            with open(self.data_file, 'r') as f:
                self.training_data = json.load(f)
            print(f"Loaded {len(self.training_data)} training examples")
        except FileNotFoundError:
            messagebox.showerror("Error", f"Could not find {self.data_file}")
            self.root.quit()
    
    def create_widgets(self):
        """Create the GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Image and navigation
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Image display
        self.image_label = ttk.Label(left_panel, text="Image will appear here", font=("Arial", 12))
        self.image_label.pack(pady=10)
        
        # Navigation buttons
        nav_frame = ttk.Frame(left_panel)
        nav_frame.pack(pady=10)
        
        ttk.Button(nav_frame, text="← Previous", command=self.previous_entry, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next →", command=self.next_entry, width=15).pack(side=tk.LEFT, padx=5)
        
        # Progress info
        self.progress_label = ttk.Label(left_panel, text="", font=("Arial", 11, "bold"))
        self.progress_label.pack(pady=5)
        
        # Right panel - Annotation form
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))
        
        # Task info
        info_frame = ttk.LabelFrame(right_panel, text="Task Information")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.task_type_label = ttk.Label(info_frame, text="", font=("Arial", 12, "bold"))
        self.task_type_label.pack(pady=5)
        
        self.prompt_text = tk.Text(info_frame, height=6, wrap=tk.WORD, font=("Arial", 10))
        self.prompt_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Annotation form
        annotation_frame = ttk.LabelFrame(right_panel, text="Your Annotation")
        annotation_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create annotation widgets
        self.create_annotation_widgets(annotation_frame)
        
        # Action buttons
        action_frame = ttk.Frame(right_panel)
        action_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(action_frame, text="Save Annotation", command=self.save_annotation, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Skip", command=self.skip_entry, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Save All & Exit", command=self.save_and_exit, width=20).pack(side=tk.RIGHT, padx=5)
    
    def create_annotation_widgets(self, parent):
        """Create annotation widgets for different task types"""
        
        # Create a notebook for different task types
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Safety Score tab
        safety_score_frame = ttk.Frame(self.notebook)
        self.notebook.add(safety_score_frame, text="Safety Score")
        
        ttk.Label(safety_score_frame, text="Safety Score (1-10)", font=("Arial", 14, "bold")).pack(pady=15)
        
        self.safety_score_var = tk.StringVar(value="5")
        safety_score_scale = ttk.Scale(safety_score_frame, from_=1, to=10, orient=tk.HORIZONTAL, 
                                      variable=self.safety_score_var, command=self.update_safety_score, length=400)
        safety_score_scale.pack(pady=15)
        
        self.safety_score_label = ttk.Label(safety_score_frame, text="Score: 5", font=("Arial", 16, "bold"))
        self.safety_score_label.pack(pady=10)
        
        ttk.Label(safety_score_frame, text="Reasoning:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(20, 5), padx=20)
        self.safety_reasoning_text = tk.Text(safety_score_frame, height=5, wrap=tk.WORD, font=("Arial", 11))
        self.safety_reasoning_text.pack(fill=tk.X, padx=20, pady=5)
        
        # Binary Classification tab
        binary_frame = ttk.Frame(self.notebook)
        self.notebook.add(binary_frame, text="Binary Classification")
        
        ttk.Label(binary_frame, text="Binary Classification", font=("Arial", 14, "bold")).pack(pady=15)
        
        self.classification_var = tk.StringVar(value="SAFE")
        ttk.Radiobutton(binary_frame, text="SAFE", variable=self.classification_var, value="SAFE", font=("Arial", 12)).pack(pady=8)
        ttk.Radiobutton(binary_frame, text="UNSAFE", variable=self.classification_var, value="UNSAFE", font=("Arial", 12)).pack(pady=8)
        
        confidence_frame = ttk.Frame(binary_frame)
        confidence_frame.pack(pady=20)
        ttk.Label(confidence_frame, text="Confidence:", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=20)
        self.confidence_var = tk.StringVar(value="MEDIUM")
        confidence_combo = ttk.Combobox(confidence_frame, textvariable=self.confidence_var, 
                                       values=["HIGH", "MEDIUM", "LOW"], state="readonly", width=15, font=("Arial", 11))
        confidence_combo.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(binary_frame, text="Reason:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(20, 5), padx=20)
        self.binary_reasoning_text = tk.Text(binary_frame, height=5, wrap=tk.WORD, font=("Arial", 11))
        self.binary_reasoning_text.pack(fill=tk.X, padx=20, pady=5)
        
        # Detailed Analysis tab
        detailed_frame = ttk.Frame(self.notebook)
        self.notebook.add(detailed_frame, text="Detailed Analysis")
        
        ttk.Label(detailed_frame, text="Detailed Analysis (1-10 each)", font=("Arial", 14, "bold")).pack(pady=15)
        
        self.detailed_scores = {}
        aspects = ["Pedestrian Safety", "Traffic Safety", "Lighting Safety", "Infrastructure Safety", "Crime Safety"]
        
        for aspect in aspects:
            aspect_frame = ttk.Frame(detailed_frame)
            aspect_frame.pack(fill=tk.X, pady=8, padx=20)
            
            ttk.Label(aspect_frame, text=f"{aspect}:", width=20, font=("Arial", 11)).pack(side=tk.LEFT)
            score_var = tk.StringVar(value="5")
            score_scale = ttk.Scale(aspect_frame, from_=1, to=10, orient=tk.HORIZONTAL, 
                                   variable=score_var, length=250, command=lambda v, a=aspect: self.update_detailed_score(v, a))
            score_scale.pack(side=tk.LEFT, padx=15)
            score_label = ttk.Label(aspect_frame, text="5", width=3, font=("Arial", 12, "bold"))
            score_label.pack(side=tk.LEFT, padx=5)
            
            self.detailed_scores[aspect] = {"var": score_var, "label": score_label, "scale": score_scale}
        
        ttk.Label(detailed_frame, text="Main Concerns:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(25, 5), padx=20)
        self.detailed_concerns_text = tk.Text(detailed_frame, height=3, wrap=tk.WORD, font=("Arial", 11))
        self.detailed_concerns_text.pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Label(detailed_frame, text="Strengths:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(15, 5), padx=20)
        self.detailed_strengths_text = tk.Text(detailed_frame, height=3, wrap=tk.WORD, font=("Arial", 11))
        self.detailed_strengths_text.pack(fill=tk.X, padx=20, pady=5)
        
        # Risk Assessment tab
        risk_frame = ttk.Frame(self.notebook)
        self.notebook.add(risk_frame, text="Risk Assessment")
        
        ttk.Label(risk_frame, text="Risk Assessment", font=("Arial", 14, "bold")).pack(pady=15)
        
        ttk.Label(risk_frame, text="HIGH RISK:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(10, 5), padx=20)
        self.high_risk_text = tk.Text(risk_frame, height=3, wrap=tk.WORD, font=("Arial", 11))
        self.high_risk_text.pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Label(risk_frame, text="MEDIUM RISK:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(15, 5), padx=20)
        self.medium_risk_text = tk.Text(risk_frame, height=3, wrap=tk.WORD, font=("Arial", 11))
        self.medium_risk_text.pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Label(risk_frame, text="LOW RISK:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(15, 5), padx=20)
        self.low_risk_text = tk.Text(risk_frame, height=3, wrap=tk.WORD, font=("Arial", 11))
        self.low_risk_text.pack(fill=tk.X, padx=20, pady=5)
        
        risk_level_frame = ttk.Frame(risk_frame)
        risk_level_frame.pack(pady=25, padx=20)
        ttk.Label(risk_level_frame, text="Overall Risk Level:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        self.overall_risk_var = tk.StringVar(value="MEDIUM")
        risk_combo = ttk.Combobox(risk_level_frame, textvariable=self.overall_risk_var, 
                                 values=["LOW", "MEDIUM", "HIGH"], state="readonly", width=15, font=("Arial", 11))
        risk_combo.pack(side=tk.LEFT, padx=15)
        
        ttk.Label(risk_frame, text="Primary Risk Factor:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(15, 5), padx=20)
        self.primary_risk_text = tk.Text(risk_frame, height=3, wrap=tk.WORD, font=("Arial", 11))
        self.primary_risk_text.pack(fill=tk.X, padx=20, pady=5)
    
    def update_safety_score(self, value):
        """Update safety score label"""
        score = int(float(value))
        self.safety_score_label.config(text=f"Score: {score}")
    
    def update_detailed_score(self, value, aspect):
        """Update detailed score label"""
        score = int(float(value))
        self.detailed_scores[aspect]["label"].config(text=str(score))
    
    def display_current_entry(self):
        """Display the current training entry"""
        if not self.training_data:
            return
        
        entry = self.training_data[self.current_index]
        
        # Update progress
        self.progress_label.config(text=f"Entry {self.current_index + 1} of {len(self.training_data)}")
        
        # Update task info
        self.task_type_label.config(text=f"Task Type: {entry['task_type'].replace('_', ' ').title()}")
        self.prompt_text.delete(1.0, tk.END)
        self.prompt_text.insert(1.0, entry['prompt'])
        
        # Load and display image
        self.load_image(entry['image_path'])
        
        # Show appropriate tab based on task type
        self.show_relevant_tab(entry['task_type'])
        
        # Load existing annotation if available
        self.load_existing_annotation(entry)
    
    def show_relevant_tab(self, task_type):
        """Show the relevant tab based on task type"""
        tab_mapping = {
            'safety_score': 0,
            'binary_classification': 1,
            'detailed_analysis': 2,
            'risk_assessment': 3
        }
        if task_type in tab_mapping:
            self.notebook.select(tab_mapping[task_type])
    
    def load_image(self, image_path):
        """Load and display the image"""
        try:
            if os.path.exists(image_path):
                image = Image.open(image_path)
                # Resize image to fit in the display area
                image.thumbnail((500, 400), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo  # Keep a reference
            else:
                self.image_label.config(image="", text=f"Image not found:\n{image_path}")
        except Exception as e:
            self.image_label.config(image="", text=f"Error loading image:\n{str(e)}")
    
    def load_existing_annotation(self, entry):
        """Load existing annotation if available"""
        if 'expected_response' in entry:
            # Parse existing annotation based on task type
            response = entry['expected_response']
            
            if entry['task_type'] == 'safety_score':
                lines = response.split('\n')
                for line in lines:
                    if line.startswith('Safety Score:'):
                        score = line.split(':')[1].strip()
                        self.safety_score_var.set(score)
                        self.safety_score_label.config(text=f"Score: {score}")
                    elif line.startswith('Reasoning:'):
                        reasoning = line.split(':', 1)[1].strip()
                        self.safety_reasoning_text.delete(1.0, tk.END)
                        self.safety_reasoning_text.insert(1.0, reasoning)
            
            elif entry['task_type'] == 'binary_classification':
                lines = response.split('\n')
                for line in lines:
                    if line.startswith('Classification:'):
                        classification = line.split(':')[1].strip()
                        self.classification_var.set(classification)
                    elif line.startswith('Confidence:'):
                        confidence = line.split(':')[1].strip()
                        self.confidence_var.set(confidence)
                    elif line.startswith('Reason:'):
                        reason = line.split(':', 1)[1].strip()
                        self.binary_reasoning_text.delete(1.0, tk.END)
                        self.binary_reasoning_text.insert(1.0, reason)
    
    def save_annotation(self):
        """Save the current annotation"""
        entry = self.training_data[self.current_index]
        task_type = entry['task_type']
        
        if task_type == 'safety_score':
            score = int(float(self.safety_score_var.get()))
            reasoning = self.safety_reasoning_text.get(1.0, tk.END).strip()
            response = f"Safety Score: {score}\nReasoning: {reasoning}"
        
        elif task_type == 'binary_classification':
            classification = self.classification_var.get()
            confidence = self.confidence_var.get()
            reason = self.binary_reasoning_text.get(1.0, tk.END).strip()
            response = f"Classification: {classification}\nConfidence: {confidence}\nReason: {reason}"
        
        elif task_type == 'detailed_analysis':
            scores = []
            total = 0
            for aspect, widgets in self.detailed_scores.items():
                score = int(float(widgets['var'].get()))
                scores.append(f"{aspect}: {score}")
                total += score
            
            concerns = self.detailed_concerns_text.get(1.0, tk.END).strip()
            strengths = self.detailed_strengths_text.get(1.0, tk.END).strip()
            
            response = f"{scores[0]}\n{scores[1]}\n{scores[2]}\n{scores[3]}\n{scores[4]}\n\nOverall Score: {total}/50\nMain Concerns: {concerns}\nStrengths: {strengths}"
        
        elif task_type == 'risk_assessment':
            high_risk = self.high_risk_text.get(1.0, tk.END).strip()
            medium_risk = self.medium_risk_text.get(1.0, tk.END).strip()
            low_risk = self.low_risk_text.get(1.0, tk.END).strip()
            overall_risk = self.overall_risk_var.get()
            primary_risk = self.primary_risk_text.get(1.0, tk.END).strip()
            
            response = f"HIGH RISK: {high_risk}\nMEDIUM RISK: {medium_risk}\nLOW RISK: {low_risk}\n\nOverall Risk Level: {overall_risk}\nPrimary Risk Factor: {primary_risk}"
        
        # Save the response
        entry['expected_response'] = response
        
        # Move to next entry
        self.next_entry()
    
    def skip_entry(self):
        """Skip the current entry"""
        self.next_entry()
    
    def previous_entry(self):
        """Go to previous entry"""
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_entry()
    
    def next_entry(self):
        """Go to next entry"""
        if self.current_index < len(self.training_data) - 1:
            self.current_index += 1
            self.display_current_entry()
        else:
            messagebox.showinfo("Complete", "You've reached the end of the dataset!")
    
    def save_and_exit(self):
        """Save all annotations and exit"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.training_data, f, indent=4)
            messagebox.showinfo("Success", "All annotations saved successfully!")
            self.root.quit()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")

def main():
    root = tk.Tk()
    app = ImprovedAnnotationHelper(root)
    root.mainloop()

if __name__ == "__main__":
    main()
