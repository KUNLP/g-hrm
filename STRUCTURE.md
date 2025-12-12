# G-HRM Project Structure

## Directory Overview

```
g-hrm/
├── src/                          # Core G-HRM implementation
│   ├── agents/                   # Multi-agent system
│   │   ├── __init__.py
│   │   ├── GraphAgent.py         # Main G-HRM agent (4-stage pipeline)
│   │   └── GraphReflectAgent.py  # Reflection agent
│   ├── tools/                    # Graph operations
│   │   ├── __init__.py
│   │   ├── graph_funcs.py       # Graph functions (Retrieve, Feature, Neighbor, Degree)
│   │   └── retriever.py          # Vector retrieval for semantic search
│   ├── __init__.py
│   ├── graph_prompts.py          # Prompt templates for all agents
│   └── graph_fewshots.py         # Few-shot examples
│
├── baselines/                    # Baseline implementations
│   ├── base_llm_qa.py           # Base LLM without KG
│   ├── graph_rag_qa.py          # GraphRAG baseline
│   └── text_rag_qa.py           # Text RAG baseline
│
├── evaluation/                   # Evaluation scripts
│   ├── evaluate_openai.py       # GPT-3.5 evaluation
│   ├── evaluate_claude.py       # Claude evaluation
│   ├── evaluate_gemini.py       # Gemini evaluation
│   ├── evaluate_graphcot.py    # Graph-CoT evaluation
│   ├── analyze_judgments.py     # Judgment analysis
│   ├── calculate_agreement_metrics.py  # Fleiss' Kappa calculation
│   ├── run_evaluation.py        # Main evaluation runner
│   └── config_evaluation.py     # Evaluation configuration
│
├── scripts/                      # Execution scripts
│   └── run_ghrm.py              # Main G-HRM execution script
│
├── configs/                      # Configuration files (optional)
├── data/                         # Data directory
│   ├── graph.json               # Knowledge graph
│   └── processed_data/          # Processed datasets
│       └── biomedical/           # Biomedical dataset
│           └── data.json        # Questions and answers
│
├── .gitignore                   # Git ignore rules
├── .env.example                 # Environment variables template
├── requirements.txt             # Python dependencies
├── README.md                    # Main documentation
└── STRUCTURE.md                # This file
```

## Key Components

### 1. G-HRM Pipeline (src/agents/GraphAgent.py)

The main implementation follows the 4-stage pipeline:

1. **Plan Generation** (`generate_multiple_plans_with_logging`)
   - Generates 5 independent reasoning plans
   - Each plan uses different strategies

2. **Strategic Self-Reflection** (`evaluate_and_select_plans_with_logging`)
   - Evaluates plans using:
     - Logical Coherence (60%)
     - Relevance (20%)
     - Completeness (20%)
   - Selects top 3 plans

3. **Parallel Execution** (`execute_parallel_plans_with_logging`)
   - Executes selected plans in parallel
   - Uses Thought-Execution agent pairs

4. **Tactical Self-Reflection** (`evaluate_and_select_answer_with_logging`)
   - Evaluates execution results
   - Selects final answer using majority voting or quality scoring

### 2. Graph Functions (src/tools/graph_funcs.py)

Core graph operations:
- `Retrieve[keyword]`: Node search by keyword
- `Feature[Node, feature]`: Extract node attributes
- `Neighbor[Node, relation]`: Get neighboring nodes
- `Degree[Node, relation]`: Count neighbors

### 3. Evaluation System (evaluation/)

Multi-model evaluation using:
- GPT-3.5-turbo
- Claude-3-haiku
- Gemini-1.5-flash

Calculates:
- Accuracy (Correct / Total)
- Soft Accuracy (Correct + Partially Correct / Total)
- Fleiss' Kappa (Inter-annotator agreement)

## File Naming Conventions

- Agent files: `*Agent.py`
- Tool files: `*_funcs.py`, `retriever.py`
- Evaluation files: `evaluate_*.py`, `analyze_*.py`
- Scripts: `run_*.py`

## Data Flow

1. **Input**: Question + Expected Answer
2. **Processing**: 
   - Plan generation → Strategic reflection → Parallel execution → Tactical reflection
3. **Output**: Final answer + Detailed logs

## Configuration

All paths are relative to project root. Hardcoded paths have been removed and replaced with:
- Environment variables (`.env` file)
- Command-line arguments
- Relative paths from project root

## Notes

- All personal information (API keys, user paths) has been removed
- Paths are now relative or configurable
- API keys should be set via environment variables
- See `.env.example` for required environment variables

