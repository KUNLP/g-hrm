#!/usr/bin/env python3
"""
Script to analyze judgment counts across different models and methodologies.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

def analyze_judgments():
    """Analyze judgment counts for all models and methodologies."""
    
    # Base directory containing the evaluation results
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = Path(os.path.join(project_root, "evaluation", "llm_evaluations"))
    
    # Models to analyze
    models = ["claude", "gemini", "openai"]
    
    # Methodologies to analyze
    methodologies = [
        "base_llm_qa_results_with_level",
        "graph_rag_qa_results_with_level", 
        "ours__results_with_level_fixed",
        "text_rag_qa_results_with_level"
    ]
    
    # Results storage
    results = {
        "summary": {},
        "detailed_results": {},
        "model_totals": {},
        "methodology_totals": {}
    }
    
    # Initialize counters
    for model in models:
        results["detailed_results"][model] = {}
        results["model_totals"][model] = {"Correct": 0, "Partially Correct": 0, "Incorrect": 0, "Total": 0}
        
        for methodology in methodologies:
            results["detailed_results"][model][methodology] = {
                "Correct": 0, 
                "Partially Correct": 0, 
                "Incorrect": 0, 
                "Total": 0
            }
    
    # Initialize methodology totals
    for methodology in methodologies:
        results["methodology_totals"][methodology] = {"Correct": 0, "Partially Correct": 0, "Incorrect": 0, "Total": 0}
    
    # Process each model
    for model in models:
        model_dir = base_dir / model
        
        if not model_dir.exists():
            print(f"Warning: Model directory {model_dir} does not exist")
            continue
            
        print(f"Processing {model} model...")
        
        # Process each methodology for this model
        for methodology in methodologies:
            # Construct filename - the actual format is {methodology}_{model}_evaluated.json
            filename = f"{methodology}_{model}_evaluated.json"
            filepath = model_dir / filename
            
            if not filepath.exists():
                print(f"Warning: File {filepath} does not exist")
                continue
                
            print(f"  Processing {methodology}...")
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Count judgments
                judgments = {"Correct": 0, "Partially Correct": 0, "Incorrect": 0, "Total": 0}
                
                # The data structure has evaluated_results array
                if "evaluated_results" in data:
                    for item in data["evaluated_results"]:
                        if "evaluation" in item and "judgment" in item["evaluation"]:
                            judgment = item["evaluation"]["judgment"]
                            if judgment in judgments:
                                judgments[judgment] += 1
                            judgments["Total"] += 1
                
                # Store results
                results["detailed_results"][model][methodology] = judgments
                
                # Update model totals
                for judgment_type in ["Correct", "Partially Correct", "Incorrect"]:
                    results["model_totals"][model][judgment_type] += judgments[judgment_type]
                results["model_totals"][model]["Total"] += judgments["Total"]
                
                # Update methodology totals
                for judgment_type in ["Correct", "Partially Correct", "Incorrect"]:
                    results["methodology_totals"][methodology][judgment_type] += judgments[judgment_type]
                results["methodology_totals"][methodology]["Total"] += judgments["Total"]
                
                print(f"    Found {judgments['Total']} total judgments: {judgments}")
                
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
    
    # Calculate summary statistics
    total_correct = sum(results["model_totals"][model]["Correct"] for model in models)
    total_partially_correct = sum(results["model_totals"][model]["Partially Correct"] for model in models)
    total_incorrect = sum(results["model_totals"][model]["Incorrect"] for model in models)
    total_problems = sum(results["model_totals"][model]["Total"] for model in models)
    
    results["summary"] = {
        "total_correct": total_correct,
        "total_partially_correct": total_partially_correct,
        "total_incorrect": total_incorrect,
        "total_problems": total_problems,
        "models_analyzed": models,
        "methodologies_analyzed": methodologies
    }
    
    return results

def save_results(results, output_file):
    """Save results to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")

def print_summary(results):
    """Print a summary of the results."""
    print("\n" + "="*80)
    print("JUDGMENT ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nTotal Problems Analyzed: {results['summary']['total_problems']}")
    print(f"Total Correct: {results['summary']['total_correct']}")
    print(f"Total Partially Correct: {results['summary']['total_partially_correct']}")
    print(f"Total Incorrect: {results['summary']['total_incorrect']}")
    
    print("\n" + "-"*60)
    print("BY MODEL:")
    print("-"*60)
    for model, totals in results["model_totals"].items():
        print(f"{model.upper()}:")
        print(f"  Correct: {totals['Correct']}")
        print(f"  Partially Correct: {totals['Partially Correct']}")
        print(f"  Incorrect: {totals['Incorrect']}")
        print(f"  Total: {totals['Total']}")
        print()
    
    print("-"*60)
    print("BY METHODOLOGY:")
    print("-"*60)
    for methodology, totals in results["methodology_totals"].items():
        print(f"{methodology}:")
        print(f"  Correct: {totals['Correct']}")
        print(f"  Partially Correct: {totals['Partially Correct']}")
        print(f"  Incorrect: {totals['Incorrect']}")
        print(f"  Total: {totals['Total']}")
        print()

if __name__ == "__main__":
    print("Starting judgment analysis...")
    
    # Analyze judgments
    results = analyze_judgments()
    
    # Print summary
    print_summary(results)
    
    # Save results to JSON file
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_file = os.path.join(project_root, "evaluation", "judgment_analysis_results.json")
    save_results(results, output_file)
    
    print(f"\nAnalysis complete! Results saved to {output_file}")
