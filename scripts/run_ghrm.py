#!/usr/bin/env python3
"""
G-HRM Runner
This script runs the G-HRM (Graph Hierarchical Reasoning with Multi-agents) framework
with configurable plan generation and selection
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.agents.GraphAgent import GraphAgent
from src.graph_prompts import graph_compound_prompt
from src.tools.retriever import NODE_TEXT_KEYS

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Graph Counselor Runner')
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (biomedical, amazon, legal, goodreads, dblp, maple)')
    parser.add_argument('--path', type=str, required=True,
                       help='Path to the dataset directory')
    
    # LLM configuration
    parser.add_argument('--llm_version', type=str, default='../model/Qwen2.5-7B-Instruct',
                       help='LLM model version to use')
    parser.add_argument('--openai_api_key', type=str, default=None,
                       help='OpenAI API key for GPT-4 mini (Self-Reflect-1)')
    
    # Enhanced Graph Counselor options
    parser.add_argument('--max_steps', type=int, default=8,  # 8단계로 제한
                        help='Maximum number of reasoning steps')
    parser.add_argument('--num_plans', type=int, default=5,
                       help='Number of plans to generate')
    parser.add_argument('--top_plans', type=int, default=3,
                       help='Number of top plans to execute')
    parser.add_argument('--enhanced_mode', action='store_true', default=True,
                       help='Enable enhanced mode with multi-plan generation and selection')
    parser.add_argument('--log_results', action='store_true', default=True,
                       help='Log results to file')
    parser.add_argument('--parallel_execution', action='store_true', default=True,
                       help='Enable parallel execution of plans')
    
    # Dataset execution options
    parser.add_argument('--run_all', action='store_true', default=True,
                       help='Run on entire dataset instead of just first 3 questions (default: True)')
    parser.add_argument('--max_questions', type=int, default=None,
                       help='Maximum number of questions to process (for testing)')
    parser.add_argument('--start_index', type=int, default=0,
                       help='Starting index for questions (0-based, default: 0)')
    parser.add_argument('--end_index', type=int, default=None,
                       help='Ending index for questions (exclusive, default: None for all)')
    
    # Optional arguments
    parser.add_argument('--ref_dataset', type=str, default=None,
                       help='Reference dataset for examples (defaults to dataset)')
    
    return parser.parse_args()

class MockArgs:
    """Mock arguments class for GraphAgent compatibility"""
    def __init__(self, args):
        self.max_steps = args.max_steps
        self.ref_dataset = args.ref_dataset or args.dataset
        self.llm_version = args.llm_version
        # path가 디렉토리인지 파일인지 확인하여 graph_dir과 data_path 설정
        if os.path.isfile(args.path):
            self.graph_dir = args.path
            self.data_path = os.path.dirname(args.path)
        else:
            # 디렉토리인 경우 graph.json은 루트 data 폴더에, data.json은 현재 경로에
            if 'processed_data' in args.path:
                # processed_data/biomedical 같은 경우
                root_data_dir = args.path.replace('/processed_data/', '/').replace('/biomedical', '').replace('/amazon', '').replace('/legal', '').replace('/goodreads', '').replace('/dblp', '').replace('/maple', '')
                self.graph_dir = os.path.join(root_data_dir, 'graph.json')
                self.data_path = args.path
            else:
                # 일반적인 경우
                self.graph_dir = os.path.join(args.path, 'graph.json')
                self.data_path = args.path
        
        self.dataset = args.dataset
        self.num_plans = args.num_plans
        self.top_plans = args.top_plans
        self.openai_api_key = args.openai_api_key
        self.run_all = args.run_all
        self.max_questions = args.max_questions
        
        # Additional required attributes for retriever
        self.faiss_gpu = False  # Use CPU for FAISS
        self.node_text_keys = NODE_TEXT_KEYS.get(args.dataset, {})
        self.embedder_name = 'sentence-transformers/all-MiniLM-L6-v2'  # Default embedder
        self.embed_cache = False  # Disable cache for simplicity
        self.embed_cache_dir = './cache'  # Default cache directory

def create_mock_args(args):
    """Create a MockArgs object compatible with GraphAgent"""
    return MockArgs(args)

def load_real_questions(dataset_path, dataset_name, run_all=False, max_questions=None, start_index=0, end_index=None):
    """Load real questions from the actual dataset"""
    try:
        # dataset_path가 파일인지 디렉토리인지 확인
        if os.path.isfile(dataset_path):
            # 파일인 경우: 디렉토리 경로 추출
            dataset_dir = os.path.dirname(dataset_path)
            print(f"Path is a file, using directory: {dataset_dir}")
        else:
            # 디렉토리인 경우: 그대로 사용
            dataset_dir = dataset_path
            print(f"Path is a directory: {dataset_dir}")
        
        data_file = os.path.join(dataset_dir, 'processed_data', 'biomedical', 'data.json')
        
        # 실제 파일이 존재하는지 확인
        if not os.path.exists(data_file):
            print(f"❌ ERROR: Data file not found at {data_file}")
            return []
        
        print(f"✅ Found data file at: {data_file}")
        
        if os.path.exists(data_file):
            # Handle JSONL format (one JSON object per line)
            data = []
            with open(data_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    try:
                        item = json.loads(line)
                        if isinstance(item, dict) and 'question' in item and 'answer' in item:
                            data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON at line {line_num}: {e}")
                        continue
            
            print(f"Total questions in dataset: {len(data)}")
            
            # 질문 수 결정 (start_index와 end_index 적용)
            if max_questions:
                questions_to_load = data[:max_questions]
                print(f"Running on first {max_questions} questions")
            elif run_all:
                # start_index와 end_index 적용
                if end_index is None:
                    end_index = len(data)
                questions_to_load = data[start_index:end_index]
                print(f"Running on questions {start_index} to {end_index-1} (total: {len(questions_to_load)} questions)")
            else:
                questions_to_load = data  # 기본값을 전체 실행으로 변경
                print("Running on ENTIRE dataset (default)")
            
            real_questions = []
            seen_questions = set()  # 중복 제거를 위한 set
            
            for i, item in enumerate(questions_to_load):
                question = item.get('question', '')
                answer = item.get('answer', '')
                if question and answer:
                    # 중복 제거: 이미 본 질문인지 확인
                    if question not in seen_questions:
                        seen_questions.add(question)
                        real_questions.append((question, answer))
                        if i < 3:  # 처음 3개만 상세 출력 (디버깅용)
                            print(f"Loaded real question {len(real_questions)}: {question[:100]}...")
                            print(f"Expected answer: {answer}")
                    else:
                        print(f"Warning: Duplicate question found at index {i}, skipping: {question[:50]}...")
            
            if real_questions:
                print(f"Successfully loaded {len(real_questions)} questions from {data_file}")
                return real_questions
        
        print(f"Warning: Could not load real questions from {data_file}")
        print(f"File exists: {os.path.exists(data_file)}")
        print(f"Directory contents: {os.listdir(os.path.dirname(data_file)) if os.path.exists(os.path.dirname(data_file)) else 'Directory not found'}")
        
    except Exception as e:
        print(f"Error loading real questions: {e}")
        import traceback
        traceback.print_exc()
    
    # Fallback to sample questions if real data loading fails
    print("Falling back to sample questions...")
    sample_questions = {
        'biomedical': [
            ("What compound can treat both type 2 diabetes mellitus and polycystic ovary syndrome?", "Metformin"),
            ("What disease is associated with gene BRCA1?", "Breast cancer"),
            ("What compound binds to gene TP53?", "Cisplatin")
        ],
        'amazon': [
            ("What is the price of iPhone?", "999"),
            ("What category is laptop?", "electronics"),
            ("What brand makes Samsung?", "Samsung")
        ],
        'legal': [
            ("What court decided this case?", "Supreme Court"),
            ("What is the case name?", "Brown v. Board"),
            ("What year was this decided?", "1954")
        ],
        'goodreads': [
            ("What author wrote Harry Potter?", "J.K. Rowling"),
            ("What year was Lord of the Rings published?", "1954"),
            ("What genre is science fiction?", "fiction")
        ],
        'dblp': [
            ("What author wrote this paper?", "John Smith"),
            ("What year was this published?", "2020"),
            ("What venue was this published in?", "ACL")
        ]
    }
    
    return sample_questions.get(dataset_name, [("Sample question?", "Sample answer")])

def main():
    """Main execution function"""
    args = parse_args()
    
    print("Enhanced Graph Counselor Runner")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Data Path: {args.path}")
    print(f"LLM Version: {args.llm_version}")
    print(f"Number of Plans: {args.num_plans}")
    print(f"Top Plans to Execute: {args.top_plans}")
    print(f"Enhanced Mode: {args.enhanced_mode}")
    print(f"Log Results: {args.log_results}")
    print(f"Parallel Execution: {args.parallel_execution}")
    print()
    
    # Check if dataset path exists
    if not os.path.exists(args.path):
        print(f"Error: Dataset path {args.path} does not exist!")
        return 1
    
    # Create mock args for GraphAgent
    mock_args = create_mock_args(args)
    
    # 디버깅 정보 출력
    print(f"Path argument: {args.path}")
    print(f"Graph directory: {mock_args.graph_dir}")
    print(f"Data path: {mock_args.data_path}")
    
    # Set OpenAI API key from environment variable or argument
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    elif not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Please set it via environment variable or --openai_api_key argument")
    
    try:
        # Initialize Enhanced Graph Agent
        print("Initializing Enhanced Graph Agent...")
        print(f"Mock args type: {type(mock_args)}")
        print(f"Mock args attributes: {[attr for attr in dir(mock_args) if not attr.startswith('_')]}")
        agent = GraphAgent(mock_args, graph_compound_prompt)
        
        # Load real questions from dataset
        real_questions = load_real_questions(mock_args.data_path, args.dataset, args.run_all, args.max_questions, args.start_index, args.end_index)
        
        print(f"Loaded {len(real_questions)} questions")
        print()
        
        # Process each question with sequential numbering
        total_questions = len(real_questions)
        for i, (question, expected_answer) in enumerate(real_questions, 1):
            print(f"\n{'='*80}")
            print(f"처리 중: 문제 {i}/{total_questions}")
            print(f"진행률: {i}/{total_questions} ({i/total_questions*100:.1f}%)")
            print(f"질문: {question}")
            print(f"예상 답변: {expected_answer}")
            print(f"{'='*80}")
            
            try:
                # Run Enhanced Graph Counselor
                agent.run(question, expected_answer)
                
                # 결과를 바로 파일에 저장
                if hasattr(agent, 'save_results_to_file'):
                    agent.save_results_to_file(question_number=f"문제 {i}")
                    print(f"✅ 문제 {i} 결과가 파일에 저장되었습니다: {agent.results_file}")
                
                print(f"Question {i} completed successfully!")
                print()
                
            except Exception as e:
                print(f"Error processing question {i}: {e}")
                # 오류가 발생해도 현재까지의 결과를 저장
                if hasattr(agent, 'save_results_to_file'):
                    try:
                        agent.save_results_to_file(question_number=f"문제 {i} (오류)")
                        print(f"⚠️ 문제 {i} 오류 결과가 파일에 저장되었습니다: {agent.results_file}")
                    except:
                        pass
                print()
                continue
        
        print("Enhanced Graph Counselor execution completed!")
        print(f"Results saved to: {agent.results_file}")
        
        return 0
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
