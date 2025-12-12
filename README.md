# G-HRM: Graph Hierarchical Reasoning with Multi-agents

Official implementation of **G-HRM (Graph Hierarchical Reasoning with Multi-agents)** - Overcoming Single-Path Brittleness in Biomedical KGQA through Parallel Multi-Agent Reasoning.

## Overview

G-HRM is a framework that addresses the single-path brittleness in Knowledge Graph Question Answering (KGQA) by combining:
1. **Divergent Thinking**: Parallel generation of multiple reasoning paths
2. **Hierarchical Convergent Thinking**: Dual self-reflection mechanisms (Strategic and Tactical) to converge to optimal answers

### Key Features

- **4-Stage Pipeline**:
  1. Plan Generation: Generate 5 independent reasoning plans
  2. Strategic Self-Reflection: Select top 3 plans based on logical coherence (60%), relevance (20%), and completeness (20%)
  3. Parallel Execution: Execute selected plans in parallel
  4. Tactical Self-Reflection: Select final answer from execution results

- **Multi-Agent Architecture**: Specialized agents for planning, execution, and reflection
- **Domain-Specific Design**: Optimized for biomedical KGQA with support for complex relationship types and synonym handling

## Installation

### Requirements

- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd g-hrm

# Create virtual environment
conda create -n g-hrm python=3.8
conda activate g-hrm

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key  # For Claude evaluation
GOOGLE_API_KEY=your-google-api-key  # For Gemini evaluation
```

## Project Structure

```
g-hrm/
├── src/                    # Core implementation
│   ├── agents/            # Agent classes
│   │   ├── GraphAgent.py          # Main G-HRM agent
│   │   └── GraphReflectAgent.py   # Reflection agent
│   ├── tools/             # Graph operations
│   │   ├── graph_funcs.py        # Graph functions (Retrieve, Feature, Neighbor, Degree)
│   │   └── retriever.py          # Vector retrieval
│   ├── graph_prompts.py   # Prompt templates
│   └── graph_fewshots.py  # Few-shot examples
├── baselines/             # Baseline implementations
│   ├── base_llm_qa.py     # Base LLM QA
│   ├── graph_rag_qa.py    # GraphRAG
│   └── text_rag_qa.py     # Text RAG
├── evaluation/            # Evaluation scripts
│   ├── evaluate_openai.py
│   ├── evaluate_claude.py
│   ├── evaluate_gemini.py
│   ├── evaluate_graphcot.py
│   └── calculate_agreement_metrics.py
├── scripts/               # Execution scripts
│   └── run_ghrm.py        # Main execution script
├── configs/               # Configuration files
├── data/                  # Data directory (graph.json, processed_data/)
└── README.md
```

## Usage

### Running G-HRM

```bash
python scripts/run_ghrm.py \
    --dataset biomedical \
    --path data/processed_data/biomedical \
    --llm_version gpt-4o-mini \
    --num_plans 5 \
    --top_plans 3 \
    --max_steps 8
```

### Arguments

- `--dataset`: Dataset name (biomedical, amazon, legal, goodreads, dblp, maple)
- `--path`: Path to dataset directory
- `--llm_version`: LLM model version (default: gpt-4o-mini)
- `--num_plans`: Number of plans to generate (default: 5)
- `--top_plans`: Number of top plans to execute (default: 3)
- `--max_steps`: Maximum reasoning steps (default: 8)
- `--openai_api_key`: OpenAI API key (optional if set in .env)

### Running Baselines

```bash
# Base LLM
python baselines/base_llm_qa.py --dataset biomedical --path data/processed_data/biomedical

# GraphRAG
python baselines/graph_rag_qa.py --dataset biomedical --path data/processed_data/biomedical

# Text RAG
python baselines/text_rag_qa.py --dataset biomedical --path data/processed_data/biomedical
```

## Evaluation

### Running Evaluations

```bash
# Evaluate with OpenAI GPT-3.5
python evaluation/evaluate_openai.py \
    --input_file results/ghrm_results.jsonl \
    --output_file results/ghrm_openai_evaluation.json

# Evaluate with Claude
python evaluation/evaluate_claude.py \
    --input_file results/ghrm_results.jsonl \
    --output_file results/ghrm_claude_evaluation.json

# Evaluate with Gemini
python evaluation/evaluate_gemini.py \
    --input_file results/ghrm_results.jsonl \
    --output_file results/ghrm_gemini_evaluation.json
```

### Calculate Agreement Metrics

```bash
python evaluation/calculate_agreement_metrics.py
```


## Citation

If you use this code, please cite:

```bibtex
@article{ghrm2025,
  title={Overcoming Single-Path Brittleness in Biomedical KGQA through Parallel Multi-Agent Reasoning},
  author={Anonymous},
  journal={HCLT 2025},
  year={2025}
}
```

## License

[Add license information]

## Contact

[Add contact information]

