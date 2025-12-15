#!/usr/bin/env python3
"""
Script to prepare evaluation datasets from baseline test data.
Extracts 30 examples from each: IndustryOR, MAMO easy, and optmath.
Converts them to the messages format for evaluation during training.
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any


def convert_to_messages_format(item: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """
    Convert a single evaluation example to messages format.
    
    Args:
        item: Dictionary containing the evaluation example
        dataset_name: Name of the dataset (for tracking)
        
    Returns:
        Dictionary in messages format
    """
    # Handle different data formats
    if "description" in item:
        # IndustryOR or similar format
        description = item["description"]
        ground_truth = item.get("ground_truth", "")
        correct_program = item.get("correct_program", "")
    elif "question" in item:
        # optmath format
        description = item["question"]
        ground_truth = item.get("ground_truth", item.get("answer", ""))
        correct_program = ""  # optmath doesn't have programs
    else:
        raise ValueError(f"Unknown data format in {dataset_name}")
    
    messages = [
        {
            "role": "system",
            "content": "You are an expert in mathematical optimization and operations research. You help solve complex optimization problems by providing clear, correct Python code using optimization libraries like Gurobi, OR-Tools, or SciPy."
        },
        {
            "role": "user",
            "content": f"Please solve this optimization problem:\n\n{description}"
        },
        {
            "role": "assistant",
            "content": f"Here's the solution to this optimization problem:\n\n```python\n{correct_program}\n```\n\nThe optimal value is: {ground_truth}" if correct_program else f"The optimal value is: {ground_truth}"
        }
    ]
    
    converted_item = {
        "messages": messages,
        "task_id": item.get("task_id", item.get("index", "")),
        "ground_truth": ground_truth,
        "dataset": dataset_name,
        "description": description,
        "correct_program": correct_program if correct_program else None
    }
    
    return converted_item


def prepare_eval_dataset(input_file: str, output_file: str, num_samples: int = 30, seed: int = 42):
    """
    Prepare evaluation dataset by sampling examples from input file.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        num_samples: Number of samples to extract
        seed: Random seed for reproducibility
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Sample examples
    if len(data) < num_samples:
        print(f"Warning: Only {len(data)} examples available, using all of them")
        sampled_data = data
    else:
        random.seed(seed)
        sampled_data = random.sample(data, num_samples)
    
    # Convert to messages format
    dataset_name = Path(input_file).stem
    converted_data = [convert_to_messages_format(item, dataset_name) for item in sampled_data]
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Prepared {len(converted_data)} examples from {input_file} -> {output_file}")


def combine_eval_datasets(dataset_files: List[str], output_file: str):
    """
    Combine multiple evaluation datasets into one file.
    
    Args:
        dataset_files: List of paths to evaluation dataset files
        output_file: Path to combined output file
    """
    combined_data = []
    
    for dataset_file in dataset_files:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            combined_data.extend(data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Combined {len(combined_data)} examples -> {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare evaluation datasets for training")
    parser.add_argument("--industryor_file", type=str, 
                       default="baseline_test_data/industryor_test.json",
                       help="Path to IndustryOR test file")
    parser.add_argument("--mamo_easy_file", type=str,
                       default="baseline_test_data/mamo_easy.json",
                       help="Path to MAMO easy test file")
    parser.add_argument("--optmath_file", type=str,
                       default="baseline_test_data/optmath_bench.json",
                       help="Path to optmath bench file")
    parser.add_argument("--output_dir", type=str,
                       default="train_test_data/eval_datasets",
                       help="Output directory for evaluation datasets")
    parser.add_argument("--num_samples", type=int, default=30,
                       help="Number of samples to extract from each dataset")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling")
    parser.add_argument("--combine", action="store_true",
                       help="Also create a combined evaluation dataset")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare individual datasets
    datasets = {
        "industryor": args.industryor_file,
        "mamo_easy": args.mamo_easy_file,
        "optmath": args.optmath_file
    }
    
    prepared_files = []
    for name, input_file in datasets.items():
        if not Path(input_file).exists():
            print(f"⚠️  Warning: {input_file} not found, skipping {name}")
            continue
        
        # Use consistent naming: industryor_eval_30.json, mamo_easy_eval_30.json, optmath_eval_30.json
        output_file = output_dir / f"{name}_eval_{args.num_samples}.json"
        prepare_eval_dataset(input_file, str(output_file), args.num_samples, args.seed)
        prepared_files.append(str(output_file))
    
    # Create combined dataset if requested
    if args.combine:
        combined_file = output_dir / f"combined_eval_{args.num_samples * len(prepared_files)}.json"
        combine_eval_datasets(prepared_files, str(combined_file))
        print(f"   Combined dataset: {combined_file}")
    
    print(f"\n✅ Evaluation datasets prepared in {output_dir}")
    print(f"   Individual datasets: {', '.join(prepared_files)}")


if __name__ == "__main__":
    main()

