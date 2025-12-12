#!/usr/bin/env python3
"""
Calculate Agreement Ratio and Fleiss' Kappa for LLM evaluation judgments
across Claude, Gemini, and OpenAI models.
"""

import json
import os
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd

def load_evaluation_data(file_path: str) -> Dict[str, str]:
    """Load evaluation data from a JSON file and extract judgments by identifier."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    judgments = {}
    
    # Handle different file structures
    if 'evaluated_results' in data:
        results = data['evaluated_results']
    elif 'evaluations' in data:
        results = data['evaluations']
    else:
        raise ValueError(f"Unknown file structure in {file_path}")
    ㄴ
    for result in results:
        # Handle different file formats - some have 'qid', others use 'question' as identifier
        if 'qid' in result:
            identifier = result['qid']
        else:
            identifier = result['question']  # Use question as identifier for GraphCoT files
        
        judgment = result['evaluation']['judgment']
        judgments[identifier] = judgment
    
    return judgments

def calculate_agreement_ratio(judgments_matrix: np.ndarray) -> float:
    """
    Calculate Agreement Ratio: proportion of items where all raters agree.
    
    Args:
        judgments_matrix: numpy array of shape (n_items, n_raters)
    
    Returns:
        Agreement ratio as a float between 0 and 1
    """
    n_items, n_raters = judgments_matrix.shape
    
    # Count items where all raters agree
    agreed_items = 0
    for i in range(n_items):
        if len(set(judgments_matrix[i])) == 1:  # All raters gave same judgment
            agreed_items += 1
    
    return agreed_items / n_items

def calculate_fleiss_kappa(judgments_matrix: np.ndarray, categories: List[str]) -> float:
    """
    Calculate Fleiss' Kappa coefficient for inter-rater agreement.
    
    Args:
        judgments_matrix: numpy array of shape (n_items, n_raters)
        categories: list of possible judgment categories
    
    Returns:
        Fleiss' Kappa coefficient
    """
    n_items, n_raters = judgments_matrix.shape
    n_categories = len(categories)
    
    # Create category mapping
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    # Count judgments per category per item
    n_ij = np.zeros((n_items, n_categories))
    for i in range(n_items):
        for j in range(n_raters):
            category = judgments_matrix[i, j]
            if category in cat_to_idx:
                n_ij[i, cat_to_idx[category]] += 1
    
    # Calculate P_j (proportion of judgments in category j)
    P_j = np.sum(n_ij, axis=0) / (n_items * n_raters)
    
    # Calculate P_i (proportion of agreement for item i)
    P_i = np.zeros(n_items)
    for i in range(n_items):
        numerator = np.sum(n_ij[i] * (n_ij[i] - 1))
        denominator = n_raters * (n_raters - 1)
        P_i[i] = numerator / denominator if denominator > 0 else 0
    
    # Calculate overall agreement
    P_bar = np.mean(P_i)
    
    # Calculate expected agreement by chance
    P_e = np.sum(P_j ** 2)
    
    # Calculate Fleiss' Kappa
    if P_e == 1:
        kappa = 1.0  # Perfect agreement
    else:
        kappa = (P_bar - P_e) / (1 - P_e)
    
    return kappa

def main():
    # Define file paths for each model
    # Use relative path from project root
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.join(project_root, "evaluation", "llm_evaluations")
    
    file_groups = {
        "base_llm_qa": {
            "claude": f"{base_dir}/claude/base_llm_qa_results_with_level_claude_evaluated.json",
            "gemini": f"{base_dir}/gemini/base_llm_qa_results_with_level_gemini_evaluated.json",
            "openai": f"{base_dir}/openai/base_llm_qa_results_with_level_openai_evaluated.json"
        },
        "graph_rag_qa": {
            "claude": f"{base_dir}/claude/graph_rag_qa_results_with_level_claude_evaluated.json",
            "gemini": f"{base_dir}/gemini/graph_rag_qa_results_with_level_gemini_evaluated.json",
            "openai": f"{base_dir}/openai/graph_rag_qa_results_with_level_openai_evaluated.json"
        },
        "graphcot": {
            "claude": f"{base_dir}/claude/graphcot_claude_evaluation.json",
            "gemini": f"{base_dir}/gemini/graphcot_gemini_evaluation.json",
            "openai": f"{base_dir}/openai/graphcot_gpt_evaluation_.json"
        },
        "ours": {
            "claude": f"{base_dir}/claude/ours__results_with_level_fixed_claude_evaluated.json",
            "gemini": f"{base_dir}/gemini/ours__results_with_level_fixed_gemini_evaluated.json",
            "openai": f"{base_dir}/openai/ours__results_with_level_fixed_openai_evaluated.json"
        },
        "text_rag_qa": {
            "claude": f"{base_dir}/claude/text_rag_qa_results_with_level_claude_evaluated.json",
            "gemini": f"{base_dir}/gemini/text_rag_qa_results_with_level_gemini_evaluated.json",
            "openai": f"{base_dir}/openai/text_rag_qa_results_with_level_openai_evaluated.json"
        }
    }
    
    categories = ["Correct", "Partially Correct", "Incorrect"]
    models = ["claude", "gemini", "openai"]
    
    results = {}
    
    print("=" * 80)
    print("AGREEMENT ANALYSIS RESULTS")
    print("=" * 80)
    
    for group_name, file_paths in file_groups.items():
        print(f"\n{group_name.upper()} EVALUATION:")
        print("-" * 50)
        
        # Load judgments for all three models
        all_judgments = {}
        for model in models:
            if os.path.exists(file_paths[model]):
                judgments = load_evaluation_data(file_paths[model])
                all_judgments[model] = judgments
                print(f"Loaded {len(judgments)} judgments from {model}")
            else:
                print(f"Warning: File not found: {file_paths[model]}")
        
        if len(all_judgments) != 3:
            print(f"Skipping {group_name} - not all three models have data")
            continue
        
        # Find common qids across all models
        common_qids = set.intersection(*[set(judgments.keys()) for judgments in all_judgments.values()])
        print(f"Common questions across all models: {len(common_qids)}")
        
        if len(common_qids) == 0:
            print(f"No common questions found for {group_name}")
            continue
        
        # Create judgments matrix
        judgments_matrix = []
        for qid in sorted(common_qids):
            row = [all_judgments[model][qid] for model in models]
            judgments_matrix.append(row)
        
        judgments_matrix = np.array(judgments_matrix)
        
        # Calculate Agreement Ratio
        agreement_ratio = calculate_agreement_ratio(judgments_matrix)
        
        # Calculate Fleiss' Kappa
        fleiss_kappa = calculate_fleiss_kappa(judgments_matrix, categories)
        
        # Store results
        results[group_name] = {
            'agreement_ratio': agreement_ratio,
            'fleiss_kappa': fleiss_kappa,
            'n_items': len(common_qids)
        }
        
        print(f"Agreement Ratio: {agreement_ratio:.4f}")
        print(f"Fleiss' Kappa: {fleiss_kappa:.4f}")
        
        # Show agreement interpretation
        if fleiss_kappa < 0:
            kappa_interp = "Poor agreement (worse than chance)"
        elif fleiss_kappa < 0.20:
            kappa_interp = "Slight agreement"
        elif fleiss_kappa < 0.40:
            kappa_interp = "Fair agreement"
        elif fleiss_kappa < 0.60:
            kappa_interp = "Moderate agreement"
        elif fleiss_kappa < 0.80:
            kappa_interp = "Substantial agreement"
        else:
            kappa_interp = "Almost perfect agreement"
        
        print(f"Kappa Interpretation: {kappa_interp}")
        
        # Show detailed agreement breakdown
        print("\nDetailed Agreement Breakdown:")
        agreement_counts = defaultdict(int)
        for row in judgments_matrix:
            if len(set(row)) == 1:
                agreement_counts[row[0]] += 1
        
        total_agreed = sum(agreement_counts.values())
        for category in categories:
            count = agreement_counts[category]
            percentage = (count / len(common_qids)) * 100
            print(f"  {category}: {count}/{len(common_qids)} ({percentage:.1f}%)")
        
        disagreement_count = len(common_qids) - total_agreed
        disagreement_percentage = (disagreement_count / len(common_qids)) * 100
        print(f"  Disagreement: {disagreement_count}/{len(common_qids)} ({disagreement_percentage:.1f}%)")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Method':<20} {'Agreement Ratio':<15} {'Fleiss Kappa':<15} {'N Items':<10}")
    print("-" * 80)
    
    for group_name, metrics in results.items():
        print(f"{group_name:<20} {metrics['agreement_ratio']:<15.4f} {metrics['fleiss_kappa']:<15.4f} {metrics['n_items']:<10}")
    
    # Overall statistics
    if results:
        avg_agreement = np.mean([m['agreement_ratio'] for m in results.values()])
        avg_kappa = np.mean([m['fleiss_kappa'] for m in results.values()])
        total_items = sum([m['n_items'] for m in results.values()])
        
        print("-" * 80)
        print(f"{'AVERAGE':<20} {avg_agreement:<15.4f} {avg_kappa:<15.4f} {total_items:<10}")
        
        print(f"\nOverall Average Agreement Ratio: {avg_agreement:.4f}")
        print(f"Overall Average Fleiss' Kappa: {avg_kappa:.4f}")
        
        if avg_kappa < 0:
            overall_interp = "Poor agreement (worse than chance)"
        elif avg_kappa < 0.20:
            overall_interp = "Slight agreement"
        elif avg_kappa < 0.40:
            overall_interp = "Fair agreement"
        elif avg_kappa < 0.60:
            overall_interp = "Moderate agreement"
        elif avg_kappa < 0.80:
            overall_interp = "Substantial agreement"
        else:
            overall_interp = "Almost perfect agreement"
        
        print(f"Overall Kappa Interpretation: {overall_interp}")

if __name__ == "__main__":
    main()
