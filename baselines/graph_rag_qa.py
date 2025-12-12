#!/usr/bin/env python3
"""
Graph RAG QA Experiment
그래프 기반 RAG로 biomedical QA를 수행하는 실험
"""

import json
import os
import time
import pickle
from datetime import datetime
from openai import OpenAI
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import networkx as nx

class GraphRAGQA:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.graph = None
        self.node_embeddings = {}
        self.node_metadata = {}
        
    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """데이터 로드"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def load_graph(self, graph_path: str):
        """그래프 데이터 로드"""
        print(f"Loading graph from {graph_path}")
        
        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        # NetworkX 그래프 생성
        self.graph = nx.Graph()
        
        # 노드 추가
        for node in graph_data.get('nodes', []):
            node_id = node.get('id')
            node_type = node.get('type', 'Unknown')
            node_name = node.get('name', '')
            
            self.graph.add_node(node_id, type=node_type, name=node_name)
            self.node_metadata[node_id] = {
                'type': node_type,
                'name': node_name,
                'id': node_id
            }
        
        # 엣지 추가
        for edge in graph_data.get('edges', []):
            source = edge.get('source')
            target = edge.get('target')
            edge_type = edge.get('type', 'Unknown')
            
            if source and target:
                self.graph.add_edge(source, target, type=edge_type)
        
        print(f"Graph loaded with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def create_node_embeddings(self):
        """노드 임베딩 생성"""
        print("Creating node embeddings...")
        
        for node_id, metadata in self.node_metadata.items():
            # 노드 정보를 텍스트로 변환
            node_text = f"Type: {metadata['type']}, Name: {metadata['name']}"
            
            # 임베딩 생성
            embedding = self.embedding_model.encode([node_text])[0]
            self.node_embeddings[node_id] = embedding
        
        print(f"Created embeddings for {len(self.node_embeddings)} nodes")
    
    def search_relevant_nodes(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """쿼리와 관련된 노드 검색"""
        # 쿼리 임베딩
        query_embedding = self.embedding_model.encode([query])[0]
        
        # 모든 노드와의 유사도 계산
        similarities = []
        for node_id, embedding in self.node_embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
            similarities.append((node_id, similarity))
        
        # 유사도 순으로 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 k개 노드 반환
        results = []
        for i, (node_id, similarity) in enumerate(similarities[:top_k]):
            metadata = self.node_metadata[node_id]
            results.append({
                "rank": i + 1,
                "node_id": node_id,
                "similarity": float(similarity),
                "type": metadata['type'],
                "name": metadata['name']
            })
        
        return results
    
    def get_neighbor_context(self, node_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """노드의 이웃 컨텍스트 가져오기"""
        context = []
        
        # BFS로 이웃 탐색
        visited = set()
        queue = [(node_id, 0)]  # (node_id, depth)
        
        while queue:
            current_node, depth = queue.pop(0)
            
            if current_node in visited or depth > max_depth:
                continue
            
            visited.add(current_node)
            
            # 현재 노드 정보 추가
            if current_node in self.node_metadata:
                metadata = self.node_metadata[current_node]
                context.append({
                    "node_id": current_node,
                    "type": metadata['type'],
                    "name": metadata['name'],
                    "depth": depth
                })
            
            # 이웃 노드 추가
            if depth < max_depth:
                neighbors = list(self.graph.neighbors(current_node))
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
        
        return context
    
    def generate_answer(self, question: str, relevant_nodes: List[Dict[str, Any]]) -> str:
        """Graph RAG 기반 답변 생성"""
        try:
            # 관련 노드들의 컨텍스트 수집
            all_contexts = []
            
            for node_info in relevant_nodes[:5]:  # 상위 5개 노드만 사용
                node_id = node_info['node_id']
                neighbor_context = self.get_neighbor_context(node_id, max_depth=1)
                
                context_text = f"Node: {node_info['name']} (Type: {node_info['type']})\n"
                context_text += f"Similarity: {node_info['similarity']:.3f}\n"
                
                if neighbor_context:
                    context_text += "Related entities:\n"
                    for neighbor in neighbor_context[:3]:  # 상위 3개 이웃만
                        if neighbor['node_id'] != node_id:
                            context_text += f"- {neighbor['name']} (Type: {neighbor['type']})\n"
                
                all_contexts.append(context_text)
            
            # 컨텍스트 구성
            context_text = "\n\n".join(all_contexts)
            
            prompt = f"""You are a medical expert. Use the following biomedical knowledge graph information to answer the question.

Knowledge Graph Context:
{context_text}

Question: {question}

Please provide a clear and accurate answer based on the provided graph information. If the information is not sufficient, say so clearly:"""
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a medical expert. Answer questions based on the provided biomedical knowledge graph information."},
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
    
    def run_experiment(self, data_path: str, graph_path: str, output_path: str, max_samples: int = None):
        """실험 실행"""
        print(f"Loading data from {data_path}")
        data = self.load_data(data_path)
        
        if max_samples:
            data = data[:max_samples]
        
        # 그래프 로드 및 임베딩 생성
        self.load_graph(graph_path)
        self.create_node_embeddings()
        
        results = []
        total_time = 0
        
        print(f"Starting Graph RAG QA experiment with {len(data)} samples")
        print(f"Model: {self.model_name}")
        print("=" * 80)
        
        for i, item in enumerate(data):
            question = item["question"]
            expected_answer = item["answer"]
            qid = item.get("qid", str(i))
            
            print(f"\n[{i+1}/{len(data)}] QID: {qid}")
            print(f"Question: {question}")
            print(f"Expected: {expected_answer}")
            
            # 관련 노드 검색
            start_time = time.time()
            relevant_nodes = self.search_relevant_nodes(question, top_k=5)
            
            # 답변 생성
            predicted_answer = self.generate_answer(question, relevant_nodes)
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
                "partial_score": evaluation["partial_score"],
                "relevant_nodes": [
                    {
                        "rank": node["rank"],
                        "node_id": node["node_id"],
                        "similarity": node["similarity"],
                        "type": node["type"],
                        "name": node["name"]
                    }
                    for node in relevant_nodes
                ]
            }
            
            results.append(result)
            
            print(f"Predicted: {predicted_answer}")
            print(f"Exact Match: {evaluation['exact_match']}")
            print(f"Partial Score: {evaluation['partial_score']:.3f}")
            print(f"Inference Time: {inference_time:.2f}s")
            print(f"Found {len(relevant_nodes)} relevant nodes")
            print("-" * 60)
        
        # 전체 통계
        exact_matches = sum(1 for r in results if r["exact_match"])
        avg_partial_score = sum(r["partial_score"] for r in results) / len(results)
        
        summary = {
            "experiment_info": {
                "model": self.model_name,
                "embedding_model": "all-MiniLM-L6-v2",
                "graph_nodes": self.graph.number_of_nodes(),
                "graph_edges": self.graph.number_of_edges(),
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
        print(f"Embedding Model: all-MiniLM-L6-v2")
        print(f"Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        print(f"Total Samples: {len(results)}")
        print(f"Exact Match Rate: {exact_matches/len(results):.3f} ({exact_matches}/{len(results)})")
        print(f"Average Partial Score: {avg_partial_score:.3f}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Average Time per Sample: {total_time/len(results):.2f}s")
        print(f"Results saved to: {output_path}")

def main():
    # 설정
    data_path = "../data/processed_data/biomedical/data.json"
    graph_path = "../data/graph.json"
    output_path = "../results/graph_rag_qa_results.json"
    
    # 결과 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 실험 실행
    qa_system = GraphRAGQA(model_name="gpt-4o-mini")
    qa_system.run_experiment(data_path, graph_path, output_path, max_samples=50)  # 테스트용 50개 샘플

if __name__ == "__main__":
    main()
