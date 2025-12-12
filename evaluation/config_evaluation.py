# Configuration for LLM evaluation
import os

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Files to evaluate
FILES_TO_EVALUATE = [
    "results_base/base_llm_qa_results_with_level.json",
    "results_base/graph_rag_qa_results_with_level.json",
    "results_base/ours__results_with_level_fixed.json",
    "results_base/text_rag_qa_results_with_level.json",
]

# Output directory
OUTPUT_DIR = "llm_evaluations"

# Evaluation settings
MAX_SAMPLES_PER_FILE = 5  # Set to None to evaluate all samples
RATE_LIMIT_DELAY = 0.5  # Delay between API calls in seconds

# LLM Models
LLM_MODELS = {
    "openai": "gpt-3.5-turbo",
    "claude": "claude-3-haiku-20240307",
    "gemini": "gemini-1.5-flash"
}
