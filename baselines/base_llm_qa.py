#!/usr/bin/env python3
"""
Base LLM QA Experiment
기본 LLM만으로 biomedical QA를 수행하는 실험
"""

import json
import os
import time
from datetime import datetime
from openai import OpenAI
from typing import List, Dict, Any

class BaseLLMQA:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        
    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """데이터 로드"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def generate_answer(self, question: str) -> str:
        """LLM으로 답변 생성"""
        try:
            prompt = f"""You are a medical expert. Please answer the following biomedical question based on your knowledge.

Question: {question}

Please provide a clear and accurate answer:"""
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a medical expert with extensive knowledge in biomedical sciences."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def evaluate_answer(self, predicted: str, expected: str) -> Dict[str, Any]:
        """답변 평가"""
        # 간단한 문자열 매칭 기반 평가
        predicted_lower = predicted.lower()
        expected_lower = expected.lower()
        
        # 정확한 매칭
        exact_match = predicted_lower == expected_lower
        
        # 부분 매칭 (예상 답변의 단어들이 예측 답변에 포함되는지)
        expected_words = set(expected_lower.split())
        predicted_words = set(predicted_lower.split())
        overlap = len(expected_words.intersection(predicted_words))
        partial_score = overlap / len(expected_words) if expected_words else 0
        
        return {
            "exact_match": exact_match,
            "partial_score": partial_score,
            "predicted": predicted,
            "expected": expected
        }
    
    def run_experiment(self, data_path: str, output_path: str, max_samples: int = None):
        """실험 실행"""
        print(f"Loading data from {data_path}")
        data = self.load_data(data_path)
        
        if max_samples:
            data = data[:max_samples]
        
        results = []
        total_time = 0
        
        print(f"Starting Base LLM QA experiment with {len(data)} samples")
        print(f"Model: {self.model_name}")
        print("=" * 80)
        
        for i, item in enumerate(data):
            question = item["question"]
            expected_answer = item["answer"]
            qid = item.get("qid", str(i))
            
            print(f"\n[{i+1}/{len(data)}] QID: {qid}")
            print(f"Question: {question}")
            print(f"Expected: {expected_answer}")
            
            # 답변 생성
            start_time = time.time()
            predicted_answer = self.generate_answer(question)
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # 평가
            evaluation = self.evaluate_answer(predicted_answer, expected_answer)
            
            result = {
                "qid": qid,
                "question": question,
                "expected_answer": expected_answer,
                "predicted_answer": predicted_answer,
                "inference_time": inference_time,
                "exact_match": evaluation["exact_match"],
                "partial_score": evaluation["partial_score"]
            }
            
            results.append(result)
            
            print(f"Predicted: {predicted_answer}")
            print(f"Exact Match: {evaluation['exact_match']}")
            print(f"Partial Score: {evaluation['partial_score']:.3f}")
            print(f"Inference Time: {inference_time:.2f}s")
            print("-" * 60)
        
        # 전체 통계
        exact_matches = sum(1 for r in results if r["exact_match"])
        avg_partial_score = sum(r["partial_score"] for r in results) / len(results)
        
        summary = {
            "experiment_info": {
                "model": self.model_name,
                "total_samples": len(results),
                "total_time": total_time,
                "avg_time_per_sample": total_time / len(results),
                "timestamp": datetime.now().isoformat()
            },
            "performance": {
                "exact_match_rate": exact_matches / len(results),
                "avg_partial_score": avg_partial_score,
                "exact_matches": exact_matches,
                "total_samples": len(results)
            },
            "results": results
        }
        
        # 결과 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"Model: {self.model_name}")
        print(f"Total Samples: {len(results)}")
        print(f"Exact Match Rate: {exact_matches/len(results):.3f} ({exact_matches}/{len(results)})")
        print(f"Average Partial Score: {avg_partial_score:.3f}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Average Time per Sample: {total_time/len(results):.2f}s")
        print(f"Results saved to: {output_path}")

def main():
    # 설정
    data_path = "../data/processed_data/biomedical/data.json"
    output_path = "../results/base_llm_qa_results.json"
    
    # 결과 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 실험 실행
    qa_system = BaseLLMQA(model_name="gpt-4o-mini")
    qa_system.run_experiment(data_path, output_path, max_samples=50)  # 테스트용 50개 샘플

if __name__ == "__main__":
    main()
