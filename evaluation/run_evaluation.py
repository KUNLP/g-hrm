#!/usr/bin/env python3
"""
Script to run LLM evaluation on biomedical QA results.
"""

import asyncio
import os
import sys
from evaluate_results_with_llms import LLMEvaluator
from config_evaluation import *


def check_environment():
    """Check if all required environment variables are set"""
    missing_vars = []

    if not OPENAI_API_KEY:
        missing_vars.append("OPENAI_API_KEY")
    if not ANTHROPIC_API_KEY:
        missing_vars.append("ANTHROPIC_API_KEY")
    if not GOOGLE_API_KEY:
        missing_vars.append("GOOGLE_API_KEY")

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these environment variables or create a .env file with:")
        print("OPENAI_API_KEY=your_openai_key")
        print("ANTHROPIC_API_KEY=your_anthropic_key")
        print("GOOGLE_API_KEY=your_google_key")
        return False

    print("✅ All required environment variables are set")
    return True


def check_files():
    """Check if all required files exist"""
    missing_files = []

    for file_path in FILES_TO_EVALUATE:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False

    print("✅ All required files found")
    return True


async def main():
    """Main function"""
    print("🚀 Starting LLM Evaluation of Biomedical QA Results")
    print("=" * 60)

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Check files
    if not check_files():
        sys.exit(1)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")

    # Initialize evaluator
    evaluator = LLMEvaluator()

    print(f"\n📊 Evaluation Configuration:")
    print(f"   - Max samples per file: {MAX_SAMPLES_PER_FILE or 'All'}")
    print(f"   - Rate limit delay: {RATE_LIMIT_DELAY}s")
    print(f"   - LLM Models: {', '.join(LLM_MODELS.values())}")

    print(f"\n📋 Files to evaluate:")
    for file_path in FILES_TO_EVALUATE:
        print(f"   - {file_path}")

    print("\n" + "=" * 60)

    # Evaluate each file
    for file_path in FILES_TO_EVALUATE:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = f"{OUTPUT_DIR}/{base_name}_llm_evaluated.json"

        print(f"\n🔍 Evaluating: {file_path}")
        print(f"💾 Output: {output_path}")

        try:
            await evaluator.evaluate_file(file_path, output_path, MAX_SAMPLES_PER_FILE)
            print(f"✅ Completed: {file_path}")
        except Exception as e:
            print(f"❌ Error evaluating {file_path}: {str(e)}")

    print("\n" + "=" * 60)
    print("🎉 Evaluation completed!")
    print(f"📁 Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
