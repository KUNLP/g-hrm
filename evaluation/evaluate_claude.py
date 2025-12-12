import json
import os
import asyncio
import aiohttp
import re
from typing import Dict, List, Any
import time
from datetime import datetime
import logging
# API Keys - Load from environment variables
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set. Please set it in your .env file or environment.")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ClaudeEvaluator:
    def __init__(self):
        self.evaluation_prompt = """# Instruction
You are an AI assistant evaluating the correctness of a model's answer for a Question-Answering task in the biomedical domain.
Compare the "Model's Answer" with the "Ground Truth Answer" and evaluate it based on the criteria below.
Provide your evaluation in a structured JSON format.

# Criteria
1.  **judgment**: Choose one from ["Correct", "Partially Correct", "Incorrect"].
    -   "Correct": The model's answer is semantically identical to the ground truth and provides the exact specific information requested.
    -   "Partially Correct": Use this judgment ONLY when the answer contains SYNONYMS or DIRECT EQUIVALENTS of the ground truth (e.g., "vitiligo" vs "leukoderma", "hypertension" vs "high blood pressure"). The answer must be contextually perfect and directly address the specific question asked. For specific entities (compounds, diseases, genes, etc.), the answer must contain SYNONYMS or DIRECT EQUIVALENTS of the exact same entity. ANY other semantic differences, missing information, additional unrelated information, or different specific entities should result in "Incorrect".
    -   "Incorrect": The answer is factually wrong, completely fails to answer the question, provides information that contradicts the ground truth, or contains NO expressions that convey the same meaning as the ground truth.
2.  **rationale**: Briefly explain the reason for your judgment based on accuracy, completeness, and specificity. For "Partially Correct", specify which expressions match the ground truth meaning.

# Important Guidelines
- For questions asking about specific cellular components, anatomical structures, or biological entities, the answer must contain the EXACT SAME TERM or its SYNONYM/DIRECT EQUIVALENT.
- For questions asking about specific compounds, diseases, genes, or other named entities, the answer must contain the EXACT SAME ENTITY or its SYNONYM/DIRECT EQUIVALENT.
- General or related answers that don't contain exact matches or synonyms should be marked as "Incorrect".
- For "Partially Correct" evaluation, the answer must contain SYNONYMS or DIRECT EQUIVALENTS of the ground truth.
- Be extremely strict: ANY semantic differences that are not synonyms, missing information, additional unrelated information, different specific entities, vague associations, broad categories, or tangentially related concepts should result in "Incorrect".
- The expression must be contextually perfect and directly relevant to the specific question asked.
- If the question asks for a specific compound and the answer provides different compounds, mark as "Incorrect" even if the compounds are related.
- If the question asks for symptoms and the answer provides different symptoms, mark as "Incorrect" even if the symptoms are related.
- If the question asks for a specific number and the answer provides a different number, mark as "Incorrect".
- Consider the specificity required by the question when making your judgment.
- Default to "Incorrect" when in doubt - only use "Partially Correct" for exact synonyms or direct equivalents.

# Examples
- Question: What cellular component is involved with most downregulated genes in disease X?
- Ground Truth Answer: nucleolus
- Model's Answer: The cellular component is the nervous system.
- Judgment: Incorrect (no semantically equivalent expression to "nucleolus")

- Question: Can you tell me about a compound that can tackle both malaria and systemic scleroderma?
- Ground Truth Answer: Pentoxifylline
- Model's Answer: Artemisinin, Dexamethasone
- Judgment: Incorrect (provides different specific compounds, not the requested compound)

- Question: What are the symptoms of atopic dermatitis?
- Ground Truth Answer: atopic dermatitis
- Model's Answer: Eczema, skin irritation, itching
- Judgment: Incorrect (provides different symptoms, not the requested disease name)

- Question: How many compounds are similar to Crotamiton?
- Ground Truth Answer: 2
- Model's Answer: Crotamiton, Benzyl benzoate, Lindane, Permethrin, Malathion
- Judgment: Incorrect (provides different number and lists compounds instead of counting)

- Question: What illness in eyelash can be alleviated by Monobenzone?
- Ground Truth Answer: vitiligo
- Model's Answer: Vitiligo
- Judgment: Correct (exact match with different capitalization)

- Question: What illness in eyelash can be alleviated by Monobenzone?
- Ground Truth Answer: vitiligo
- Model's Answer: leukoderma
- Judgment: Partially Correct (synonym of vitiligo)

- Question: What illness in eyelash can be alleviated by Monobenzone?
- Ground Truth Answer: vitiligo
- Model's Answer: Vitiligo, skin pigmentation disorder
- Judgment: Incorrect (contains additional unrelated information)

# Task
Evaluate the following:
- Question: {{QUESTION}}
- Ground Truth Answer: {{GROUND_TRUTH}}
- Model's Answer: {{MODEL_ANSWER}}

# Output (JSON format only)"""

    def clean_json_response(self, response_text):
        """Clean and extract JSON from response"""
        # Remove markdown code blocks
        response_text = re.sub(r"```json\s*", "", response_text)
        response_text = re.sub(r"```\s*$", "", response_text)

        # Try to find JSON object
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            return json_match.group(0)

        return response_text

    async def evaluate_with_claude(
        self,
        question: str,
        ground_truth: str,
        model_answer: str,
        session: aiohttp.ClientSession,
    ) -> Dict[str, Any]:
        """Evaluate using Claude Haiku"""
        try:
            prompt = (
                self.evaluation_prompt.replace("{{QUESTION}}", question)
                .replace("{{GROUND_TRUTH}}", ground_truth)
                .replace("{{MODEL_ANSWER}}", model_answer)
            )

            headers = {
                "x-api-key": ANTHROPIC_API_KEY,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }

            data = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}],
            }

            async with session.post(
                "https://api.anthropic.com/v1/messages", headers=headers, json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result["content"][0]["text"]
                    try:
                        evaluation = json.loads(content)
                        return {
                            "llm": "claude-haiku",
                            "judgment": evaluation.get("judgment", "Unknown"),
                            "rationale": evaluation.get(
                                "rationale", "No rationale provided"
                            ),
                            "raw_response": content,
                        }
                    except json.JSONDecodeError:
                        return {
                            "llm": "claude-haiku",
                            "judgment": "Error",
                            "rationale": "Failed to parse JSON response",
                            "raw_response": content,
                        }
                else:
                    return {
                        "llm": "claude-haiku",
                        "judgment": "Error",
                        "rationale": f"API error: {response.status}",
                        "raw_response": "",
                    }
        except Exception as e:
            return {
                "llm": "claude-haiku",
                "judgment": "Error",
                "rationale": f"Exception: {str(e)}",
                "raw_response": "",
            }

    async def evaluate_single_result(
        self, result: Dict[str, Any], session: aiohttp.ClientSession
    ) -> Dict[str, Any]:
        """Evaluate a single result with Claude"""
        question = result.get("question", "")
        expected_answer = result.get("expected_answer", "")
        predicted_answer = result.get("predicted_answer", "")

        evaluation = await self.evaluate_with_claude(
            question, expected_answer, predicted_answer, session
        )

        return {
            "qid": result.get("qid", ""),
            "question": question,
            "expected_answer": expected_answer,
            "predicted_answer": predicted_answer,
            "level": result.get("level", ""),
            "evaluation": evaluation,
        }

    async def evaluate_file(
        self, file_path: str, output_path: str, max_samples: int = None
    ) -> None:
        """Evaluate all results in a file"""
        logger.info(f"Starting Claude evaluation of {file_path}")

        # Load the results file
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = data.get("results", [])
        if max_samples:
            results = results[:max_samples]

        logger.info(f"Evaluating {len(results)} samples from {file_path}")

        # Create output structure
        output_data = {
            "source_file": file_path,
            "evaluation_info": {
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(results),
                "evaluator": "claude-haiku",
            },
            "evaluated_results": [],
        }

        # Evaluate results with rate limiting
        async with aiohttp.ClientSession() as session:
            for i, result in enumerate(results):
                logger.info(f"Evaluating sample {i+1}/{len(results)} from {file_path}")

                try:
                    evaluated_result = await self.evaluate_single_result(
                        result, session
                    )
                    output_data["evaluated_results"].append(evaluated_result)

                    # Add small delay to avoid rate limiting
                    await asyncio.sleep(0.5)

                except Exception as e:
                    logger.error(f"Error evaluating sample {i+1}: {str(e)}")
                    output_data["evaluated_results"].append(
                        {
                            "qid": result.get("qid", ""),
                            "question": result.get("question", ""),
                            "expected_answer": result.get("expected_answer", ""),
                            "predicted_answer": result.get("predicted_answer", ""),
                            "level": result.get("level", ""),
                            "evaluation": {
                                "llm": "claude-haiku",
                                "judgment": "Error",
                                "rationale": str(e),
                            },
                        }
                    )

        # Save results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Claude evaluation completed for {file_path}. Results saved to {output_path}"
        )


async def main():
    """Main function to evaluate all result files with Claude"""
    evaluator = ClaudeEvaluator()

    # Define the files to evaluate
    files_to_evaluate = [
        # Example path - update with your actual results path
        # "results/base_llm_qa_results.json",
    ]

    # Create output directory
    os.makedirs("llm_evaluations/claude", exist_ok=True)

    # Evaluate each file
    for file_path in files_to_evaluate:
        if os.path.exists(file_path):
            # Create output filename
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = f"llm_evaluations/claude/{base_name}_claude_evaluated.json"

            # Evaluate all samples
            await evaluator.evaluate_file(file_path, output_path, max_samples=None)
        else:
            logger.warning(f"File not found: {file_path}")


if __name__ == "__main__":
    asyncio.run(main())
