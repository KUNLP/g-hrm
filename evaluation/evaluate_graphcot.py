#!/usr/bin/env python3
"""
GraphCOT Results Evaluation Script
GraphCOT 결과를 평가하는 스크립트
"""

import json
import os
import time
from openai import OpenAI
from typing import List, Dict, Any
import argparse

class GraphCOTEvaluator:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        # API 키는 환경 변수에서 로드
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        
    def load_graphcot_results(self, file_path: str) -> List[Dict[str, Any]]:
        """GraphCOT 결과 파일 로드"""
        results = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"JSON parsing error at line {line_num}: {e}")
                        print(f"Problematic line: {line}")
                        continue
        return results
    
    def evaluate_answer(self, question: str, model_answer: str, ground_truth: str) -> Dict[str, Any]:
        """답변 평가"""
        try:
            prompt = f"""# Instruction
You are an AI assistant evaluating the correctness of a model's answer for a Question-Answering task in the biomedical domain.
Compare the "Model's Answer" with the "Ground Truth Answer" and evaluate it based on the criteria below.
Provide your evaluation in a structured JSON format.

# Criteria
1.  **judgment**: Choose one from ["Correct", "Partially Correct", "Incorrect"].
    -   "Correct": The model's answer is semantically identical to the ground truth and provides all the required information.
    -   "Partially Correct": The answer is factually correct but incomplete or contains minor irrelevant information.
    -   "Incorrect": The answer is factually wrong or completely fails to answer the question.
2.  **rationale**: Briefly explain the reason for your judgment based on accuracy, completeness, and relevance.

# Example
- Question: What are the side effects of Malathion?
- Ground Truth Answer: ["diarrhea", "cramps", "blurred vision"]
- Model's Answer: Malathion can cause diarrhea and cramps.

# Task
Evaluate the following:
- Question: {question}
- Ground Truth Answer: {ground_truth}
- Model's Answer: {model_answer}

# Output (JSON format only)"""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an AI assistant evaluating biomedical question-answering results. Always respond with valid JSON format only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            # JSON 파싱
            evaluation_text = response.choices[0].message.content.strip()
            try:
                evaluation = json.loads(evaluation_text)
                return evaluation
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 기본값 반환
                return {
                    "judgment": "Incorrect",
                    "rationale": "Failed to parse evaluation response"
                }
                
        except Exception as e:
            return {
                "judgment": "Incorrect",
                "rationale": f"Evaluation error: {str(e)}"
            }
    
    def run_evaluation(self, input_file: str, output_file: str, max_samples: int = None):
        """평가 실행"""
        print(f"Loading GraphCOT results from {input_file}")
        results = self.load_graphcot_results(input_file)
        
        if max_samples:
            results = results[:max_samples]
        
        print(f"Starting GraphCOT evaluation with {len(results)} samples")
        print(f"Model: {self.model_name}")
        print("=" * 80)
        
        evaluations = []
        total_time = 0
        
        for i, result in enumerate(results):
            question = result["question"]
            model_answer = result["model_answer"]
            ground_truth = result["gt_answer"]
            
            print(f"\n[{i+1}/{len(results)}]")
            print(f"Question: {question}")
            print(f"Model Answer: {model_answer}")
            print(f"Ground Truth: {ground_truth}")
            
            # 평가 실행
            start_time = time.time()
            evaluation = self.evaluate_answer(question, model_answer, ground_truth)
            evaluation_time = time.time() - start_time
            total_time += evaluation_time
            
            # 결과 저장
            eval_result = {
                "question": question,
                "model_answer": model_answer,
                "ground_truth": ground_truth,
                "evaluation": evaluation,
                "evaluation_time": evaluation_time
            }
            evaluations.append(eval_result)
            
            print(f"Judgment: {evaluation['judgment']}")
            print(f"Rationale: {evaluation['rationale']}")
            print(f"Evaluation Time: {evaluation_time:.2f}s")
            print("-" * 60)
        
        # 통계 계산
        correct_count = sum(1 for e in evaluations if e["evaluation"]["judgment"] == "Correct")
        partial_count = sum(1 for e in evaluations if e["evaluation"]["judgment"] == "Partially Correct")
        incorrect_count = sum(1 for e in evaluations if e["evaluation"]["judgment"] == "Incorrect")
        
        summary = {
            "experiment_info": {
                "model": self.model_name,
                "evaluation_model": self.model_name,
                "total_samples": len(evaluations),
                "total_time": total_time,
                "avg_time_per_sample": total_time / len(evaluations)
            },
            "performance": {
                "correct": correct_count,
                "partially_correct": partial_count,
                "incorrect": incorrect_count,
                "correct_rate": correct_count / len(evaluations),
                "partially_correct_rate": partial_count / len(evaluations),
                "incorrect_rate": incorrect_count / len(evaluations)
            },
            "evaluations": evaluations
        }
        
        # 결과 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Model: {self.model_name}")
        print(f"Total Samples: {len(evaluations)}")
        print(f"Correct: {correct_count} ({correct_count/len(evaluations):.3f})")
        print(f"Partially Correct: {partial_count} ({partial_count/len(evaluations):.3f})")
        print(f"Incorrect: {incorrect_count} ({incorrect_count/len(evaluations):.3f})")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Average Time per Sample: {total_time/len(evaluations):.2f}s")
        print(f"Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='GraphCOT Results Evaluation')
    parser.add_argument('--input_file', type=str, 
                       default=None,
                       help='Input GraphCOT results file')
    parser.add_argument('--output_file', type=str, 
                       default='results/graphcot_evaluation_results_ver1.json',
                       help='Output evaluation results file')
    parser.add_argument('--max_samples', type=int, default=270, help='Maximum number of samples to evaluate')
    args = parser.parse_args()
    
    # 결과 디렉토리 생성
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 평가 실행
    evaluator = GraphCOTEvaluator(model_name="gpt-4o-mini")
    evaluator.run_evaluation(args.input_file, args.output_file, args.max_samples)

if __name__ == "__main__":
    main()
