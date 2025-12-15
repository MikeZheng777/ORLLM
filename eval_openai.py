import argparse
import json
import os
import re
import sys
import subprocess
import tempfile
import concurrent.futures
from collections import Counter
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from utils.gurobi import execute_gurobi, test_optimality

# OpenAI imports
try:
    import openai
    from dotenv import load_dotenv
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai or python-dotenv not installed. Install with: pip install openai python-dotenv")

TEMPLATE_q2mc_en = r"""
Below is an operations research question. First formulate the problem as an optimization problem. Then according to the formulated problem, write python code that uses 'gurobipy' to solve the optimization problem. You should write a complete python code that can be executed to solve the optimization problem.

IMPORTANT: Your code should follow these requirements:
1. Create a Gurobi model variable named 'model' (not 'm' or other names)
2. Include all necessary imports and data setup
3. Write the complete optimization code that can run independently
4. Do NOT wrap everything in a function - write the code directly
5. Make sure the model is created and optimized in the main execution flow
6. The code should print the optimal objective value when solved successfully

Example format:
```python
import gurobipy as gp
from gurobipy import GRB

# Data setup
data = {...}

# Create model
model = gp.Model("ProblemName")

# Decision variables
x = model.addVars(...)

# Objective
model.setObjective(...)

# Constraints
model.addConstr(...)

# Optimize
model.optimize()

# Results
if model.status == GRB.OPTIMAL:
    print(f"Optimal value: {model.objVal}")
else:
    print("No optimal solution found")
```

# Question:
{Question}

# Response:
"""

ONE_QUESTION = r"""
A lab has 1000 units of medicinal ingredients to make two pills, a large pill and a small pill. A large pill requires 3 units of medicinal ingredients and 2 units of filler. A small pill requires 2 units of medicinal ingredients and 1 unit of filler. The lab has to make at least 100 large pills. However, since small pills are more popular at least 60% of the total number of pills must be small. How many of each should be made to minimize the total number of filler material needed?
"""

def _run_gurobi_in_process(script_content, timeout, queue):
    """Run execute_gurobi in a separate process. Must be module-level for pickling."""
    try:
        # Set a shorter timeout for Gurobi (leave some buffer for process overhead)
        process_overhead_buffer = min(max(8, int(timeout * 0.15)), 20)
        gurobi_timeout = max(1, timeout - process_overhead_buffer)
        result = execute_gurobi(script_content, timeout=gurobi_timeout)
        queue.put(result)
    except Exception as e:
        queue.put({"success": False, "value": str(e)})

def extract_code_from_output(output):
    """Extract Python code from model output."""
    start = output.find("```python")
    if start == -1:
        return None
    end = output.find("```", start + 9)
    if end == -1:
        return None
    return output[start+9:end].strip()

def compile_script(script_content, timeout=300):
    """Execute Python script and capture results using Gurobi evaluation with timeout."""
    target_dir = './eval_execute'
    os.makedirs(target_dir, exist_ok=True)
    
    # Use multiprocessing with spawn context for robust timeout enforcement
    import multiprocessing
    import queue
    import signal
    
    ctx = multiprocessing.get_context('spawn')
    result_queue = ctx.Queue()
    
    process = ctx.Process(target=_run_gurobi_in_process, args=(script_content, timeout, result_queue))
    process.start()
    process.join(timeout=timeout)
    
    if process.is_alive():
        # Process is still running, terminate it
        try:
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        except Exception:
            pass
        try:
            os.kill(process.pid, signal.SIGKILL)
        except Exception:
            pass
        
        execution_state = "Execution Failed: Timeout"
        execution_best_solution = None
    else:
        # Process completed, get result
        try:
            result = result_queue.get(timeout=2)
        except queue.Empty:
            execution_state = "Execution Failed: No result from process"
            execution_best_solution = None
            result = {"success": False, "value": "No result"}
        
        if result.get("success", False):
            execution_state = "Execution Successful"
            execution_best_solution = result.get("value")
        else:
            execution_state = f"Execution Failed: {result.get('value', 'Unknown error')}"
            execution_best_solution = None
    
    return {
        "execution_result": execution_state,
        "execution_best_solution": execution_best_solution,
        "execution_state": execution_state
    }

def assess_code_correctness(code, ground_truth=None, numerical_tolerance=0.05, timeout=300):
    """Assess code correctness focusing on execution success and mathematical accuracy."""
    correctness_metrics = {
        "execution_success": False,
        "mathematical_accuracy": False
    }
    
    if not code:
        return correctness_metrics, {
            "execution_result": "No code provided",
            "execution_best_solution": None,
            "execution_state": "No code provided"
        }
    
    # Execute the code directly using Gurobi
    execution_output = compile_script(code, timeout=timeout)
    
    # Check execution success
    if "Execution Successful" in execution_output["execution_state"]:
        correctness_metrics["execution_success"] = True
        
        # Check mathematical accuracy using test_optimality
        if ground_truth is not None:
            try:
                if execution_output["execution_best_solution"] == "No Best Solution":
                    if str(ground_truth).lower() == "no best solution":
                        correctness_metrics["mathematical_accuracy"] = True
                else:
                    # Use Gurobi test_optimality function for mathematical accuracy
                    # Use numerical_tolerance as tolerance for relative error calculation
                    optimality_result = test_optimality(code, float(ground_truth), timeout=timeout, tolerance=numerical_tolerance)
                    if optimality_result == "correct":
                        correctness_metrics["mathematical_accuracy"] = True
            except (ValueError, TypeError, ZeroDivisionError):
                pass
    
    return correctness_metrics, execution_output

def load_test_data(test_file):
    """Load test data from JSON file."""
    if not os.path.exists(test_file):
        return []
    
    with open(test_file, 'r') as f:
        if test_file.endswith('.jsonl'):
            return [json.loads(line) for line in f]
        else:
            return json.load(f)

def generate_with_openai(client, prompt, model_name, topk=1, temperature=0, max_tokens=None):
    """Generate responses using OpenAI API."""
    try:
        # Prepare parameters
        params = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "n": topk
        }
        
        # Handle temperature - use small value for greedy decoding
        if temperature > 0:
            params["temperature"] = temperature
        else:
            # For greedy decoding, use temperature=0 or omit it
            params["temperature"] = 0
        
        # Add max_tokens/max_completion_tokens if specified
        # Newer models (like gpt-5.1) use max_completion_tokens instead of max_tokens
        if max_tokens is not None:
            # Check if model name suggests it's a newer model that requires max_completion_tokens
            if "gpt-5" in model_name.lower() or "o3" in model_name.lower():
                params["max_completion_tokens"] = max_tokens
            else:
                # Try max_tokens first for older models
                params["max_tokens"] = max_tokens
        
        try:
            response = client.chat.completions.create(**params)
        except Exception as e:
            # If max_tokens fails with error about max_completion_tokens, retry with max_completion_tokens
            error_str = str(e).lower()
            if "max_tokens" in error_str and "max_completion_tokens" in error_str:
                if "max_tokens" in params:
                    del params["max_tokens"]
                    params["max_completion_tokens"] = max_tokens
                    response = client.chat.completions.create(**params)
                else:
                    raise
            else:
                raise
        
        # Convert OpenAI response format to match vLLM format
        outputs = []
        for choice in response.choices:
            outputs.append(type('Output', (), {
                'text': choice.message.content,
                'finish_reason': choice.finish_reason
            })())
        
        return type('Generation', (), {'outputs': outputs})()
        
    except Exception as e:
        print(f"Error generating with OpenAI: {e}")
        # Return empty generation on error
        return type('Generation', (), {'outputs': []})()

def main(args):
    if not OPENAI_AVAILABLE:
        print("Error: OpenAI library not available. Install with: pip install openai python-dotenv")
        sys.exit(1)
    
    assert isinstance(args.topk, int)
    assert args.decoding_method in ["greedy", "sampling"]
    
    # Load environment variables from .env file if it exists
    if os.path.exists('.env'):
        load_dotenv()
    
    # Initialize OpenAI client
    api_key = args.api_key if hasattr(args, 'api_key') and args.api_key else os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key not found. Set OPENAI_API_KEY environment variable or use --api_key")
        sys.exit(1)
    
    client = openai.OpenAI(api_key=api_key)
    print(f"Initialized OpenAI client for model: {args.model_name}")
    
    # Load test data if provided
    test_data = []
    if args.test_file:
        test_data = load_test_data(args.test_file)
        print(f"Loaded {len(test_data)} test examples")
    
    # Prepare samples for evaluation
    if test_data:
        # Use test data
        sample = []
        for example in test_data:
            question_key = args.question_field if args.question_field else "description"
            if question_key in example:
                prompt = TEMPLATE_q2mc_en.replace("{Question}", example[question_key].strip()).strip()
                example_t = {k: v for k, v in example.items()}
                example_t["prompt"] = prompt
                sample.append(example_t)
    else:
        # Use single question
        prompt = TEMPLATE_q2mc_en.replace("{Question}", ONE_QUESTION.strip()).strip()
        sample = [{"prompt": prompt, "ground_truth": None}]
    
    # Set generation parameters
    default_max_tokens = 10000
    max_tokens_value = args.max_tokens if args.max_tokens is not None else default_max_tokens
    
    if args.decoding_method == "greedy":
        temperature = 0
    elif args.decoding_method == "sampling":
        temperature = args.temperature
    else:
        raise ValueError("Invalid decoding method")
    
    print(f"Generation parameters - topk: {args.topk}, temperature: {temperature}, max_tokens: {max_tokens_value}")
    
    # Generate responses
    prompts = [example["prompt"] for example in sample]
    generations = []
    
    print("Generating responses with OpenAI API...")
    for prompt in tqdm(prompts, desc="Generating"):
        generation = generate_with_openai(
            client, prompt, args.model_name,
            topk=args.topk, temperature=temperature,
            max_tokens=max_tokens_value
        )
        generations.append(generation)
    
    # Evaluate correctness
    results = []
    overall_metrics = {
        "total_samples": 0,
        "execution_success": 0,
        "mathematical_accuracy": 0,
        "code_extraction_success": 0
    }
    
    for idx, (example, prompt, generation) in enumerate(zip(sample, prompts, generations), 1):
        outputs = generation.outputs
        
        if args.verbose:
            print(f"\nProcessing sample {idx}/{len(sample)}")
        
        for output in outputs:
            result_entry = {k: v for k, v in example.items()}
            result_entry["generated_output"] = output.text
            
            # Extract code
            extracted_code = extract_code_from_output(output.text)
            result_entry["extracted_code"] = extracted_code
            
            if extracted_code:
                overall_metrics["code_extraction_success"] += 1
                
                # Assess correctness with timeout
                ground_truth = example.get(args.answer_field)
                if args.verbose:
                    print(f"  Executing code (timeout: {args.timeout}s)...")
                
                import time
                exec_start = time.time()
                try:
                    correctness_metrics, execution_output = assess_code_correctness(
                        extracted_code, ground_truth, args.numerical_tolerance, args.timeout)
                    exec_elapsed = time.time() - exec_start
                    if args.verbose:
                        print(f"  Code execution completed in {exec_elapsed:.1f}s")
                except Exception as e:
                    exec_elapsed = time.time() - exec_start
                    if args.verbose:
                        print(f"  Error during code execution (after {exec_elapsed:.1f}s): {e}")
                    correctness_metrics = {
                        "execution_success": False,
                        "mathematical_accuracy": False
                    }
                    execution_output = {
                        "execution_result": f"Error: {str(e)}",
                        "execution_best_solution": None,
                        "execution_state": f"Execution Failed: {str(e)}"
                    }
                
                result_entry.update(correctness_metrics)
                result_entry.update(execution_output)
                
                # Update overall metrics
                for metric in ["execution_success", "mathematical_accuracy"]:
                    if correctness_metrics[metric]:
                        overall_metrics[metric] += 1
                
                # Verbose output
                if args.verbose:
                    print(f"\n{'='*60}")
                    print(f"Sample {overall_metrics['total_samples'] + 1}")
                    print(f"Question: {example.get(args.question_field, 'N/A')[:100]}...")
                    print(f"Ground Truth: {ground_truth}")
                    print(f"Predicted Solution: {execution_output.get('execution_best_solution', 'N/A')}")
                    print(f"Correctness Metrics: {correctness_metrics}")
                    print(f"Execution State: {execution_output.get('execution_state', 'N/A')}")
                    print(f"{'='*60}")
            else:
                # No code found
                result_entry.update({
                    "execution_success": False,
                    "mathematical_accuracy": False,
                    "execution_result": "No code found",
                    "execution_best_solution": None,
                    "execution_state": "No code found"
                })
            
            overall_metrics["total_samples"] += 1
            results.append(result_entry)
    
    # Save results
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"Results saved to {args.output_file}")
    
    # Calculate and display final metrics
    final_metrics = {
        "total_samples": overall_metrics["total_samples"]
    }
    for metric in ["execution_success", "mathematical_accuracy", "code_extraction_success"]:
        if overall_metrics["total_samples"] > 0:
            final_metrics[metric] = overall_metrics[metric] / overall_metrics["total_samples"]
        else:
            final_metrics[metric] = 0.0
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total samples evaluated: {overall_metrics['total_samples']}")
    print(f"Code extraction success: {final_metrics['code_extraction_success']:.3f}")
    print(f"Execution success: {final_metrics['execution_success']:.3f}")
    print(f"Mathematical accuracy: {final_metrics['mathematical_accuracy']:.3f}")
    
    # Save metrics
    if args.output_file:
        metrics_file = args.output_file.replace('.json', '_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        print(f"Metrics saved to {metrics_file}")
    
    return final_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate optimization code generation with OpenAI API")
    parser.add_argument("--model_name", type=str, required=True, help="OpenAI model name (e.g., gpt-4o, gpt-4-turbo)")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--topk", type=int, default=1, help="Number of responses to generate per prompt")
    parser.add_argument("--decoding_method", type=str, default="greedy", choices=["greedy", "sampling"], 
                       help="Decoding method")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--max_tokens", type=int, default=None, help="Maximum tokens to generate")
    
    # Input/Output files
    parser.add_argument("--test_file", type=str, default=None, 
                       help="Path to test data file (JSON or JSONL)")
    parser.add_argument("--output_file", type=str, default=None, 
                       help="Path to save evaluation results")
    
    # Test data configuration
    parser.add_argument("--question_field", type=str, default="description", 
                       help="Field name containing questions in test data")
    parser.add_argument("--answer_field", type=str, default="ground_truth", 
                       help="Field name containing ground truth answers")
    
    # Evaluation parameters
    parser.add_argument("--timeout", type=int, default=60, 
                       help="Timeout for code execution in seconds")
    parser.add_argument("--numerical_tolerance", type=float, default=0.05, 
                       help="Tolerance for numerical accuracy comparison")
    parser.add_argument("--verbose", action="store_true", 
                       help="Print detailed evaluation information")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

