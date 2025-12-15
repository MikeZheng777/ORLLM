#!/usr/bin/env python3
"""
Script to convert optimization problem data to messages format for LLM finetuning.

This script converts the data from train_data_4o_new_100.json into a format
that can be used with the existing finetuning pipeline in train/data.py and train/finetune.py.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def convert_to_messages_format(input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert optimization problem data to messages format for finetuning.
    
    Args:
        input_data: List of dictionaries containing optimization problems
        
    Returns:
        List of dictionaries in messages format
    """
    converted_data = []
    
    for item in input_data:
        # Create a conversation-style format
        messages = [
            {
                "role": "system",
                "content": "You are an expert in mathematical optimization and operations research. You help solve complex optimization problems by providing clear, correct Python code using optimization libraries like Gurobi, OR-Tools, or SciPy."
            },
            {
                "role": "user", 
                "content": f"Please solve this optimization problem:\n\n{item['description']}"
            },
            {
                "role": "assistant",
                "content": f"Here's the solution to this optimization problem:\n\n```python\n{item['correct_program']}\n```\n\nThe optimal value is: {item['ground_truth']}"
            }
        ]
        
        converted_item = {
            "messages": messages,
            "task_id": item.get("task_id", ""),
            "ground_truth": item.get("ground_truth", ""),
            "problem_domain": item.get("tag", [{}])[0].get("problem_domain", "") if item.get("tag") else "",
            "modeling_type": item.get("tag", [{}])[0].get("modeling_type", "") if item.get("tag") else "",
            "industry_sector": item.get("tag", [{}])[0].get("industry_sector", "") if item.get("tag") else ""
        }
        
        converted_data.append(converted_item)
    
    return converted_data


def convert_to_prompt_completion_format(input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert optimization problem data to prompt/completion format for finetuning.
    
    Args:
        input_data: List of dictionaries containing optimization problems
        
    Returns:
        List of dictionaries in prompt/completion format
    """
    converted_data = []
    
    for item in input_data:
        prompt = f"Please solve this optimization problem:\n\n{item['description']}"
        
        completion = f"Here's the solution to this optimization problem:\n\n```python\n{item['correct_program']}\n```\n\nThe optimal value is: {item['ground_truth']}"
        
        converted_item = {
            "prompt": prompt,
            "completion": completion,
            "task_id": item.get("task_id", ""),
            "ground_truth": item.get("ground_truth", ""),
            "problem_domain": item.get("tag", [{}])[0].get("problem_domain", "") if item.get("tag") else "",
            "modeling_type": item.get("tag", [{}])[0].get("modeling_type", "") if item.get("tag") else "",
            "industry_sector": item.get("tag", [{}])[0].get("industry_sector", "") if item.get("tag") else ""
        }
        
        converted_data.append(converted_item)
    
    return converted_data


def main():
    parser = argparse.ArgumentParser(description="Convert optimization data for LLM finetuning")
    parser.add_argument("--input_file", type=str, required=True, 
                       help="Path to input JSON file (e.g., train_data_4o_new_100.json)")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Path to output JSON file")
    parser.add_argument("--format", type=str, choices=["messages", "prompt_completion"], 
                       default="messages", help="Output format (default: messages)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to convert (for testing)")
    
    args = parser.parse_args()
    
    # Load input data
    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    print(f"Loaded {len(input_data)} samples")
    
    # Limit samples if specified
    if args.max_samples:
        input_data = input_data[:args.max_samples]
        print(f"Limited to {len(input_data)} samples for conversion")
    
    # Convert data
    print(f"Converting to {args.format} format...")
    if args.format == "messages":
        converted_data = convert_to_messages_format(input_data)
    else:
        converted_data = convert_to_prompt_completion_format(input_data)
    
    # Save converted data
    print(f"Saving converted data to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully converted {len(converted_data)} samples to {args.format} format")
    print(f"Output saved to: {args.output_file}")
    
    # Show a sample of the converted data
    if converted_data:
        print("\nSample of converted data:")
        print(json.dumps(converted_data[0], indent=2)[:500] + "...")


if __name__ == "__main__":
    main()
