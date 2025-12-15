"""
Custom evaluation callback that runs full evaluation (like eval.py) on separate datasets.
Evaluates on IndustryOR, MAMO easy, and optmath separately and reports metrics for each.
Uses vLLM for fast inference during training.
"""

import os
import json
import re
import torch
import logging
from typing import Dict, List, Any, Optional
from transformers import TrainerCallback, TrainerState, TrainerControl
from tqdm import tqdm

# Import vLLM for fast inference
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logging.warning("vLLM not available, falling back to model.generate()")

# Import evaluation utilities from eval.py
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils.gurobi import execute_gurobi, test_optimality

logger = logging.getLogger(__name__)

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


def _run_gurobi_in_process(script_content, timeout, queue):
    """Run execute_gurobi in a separate process. Must be module-level for pickling."""
    try:
        # Set a shorter timeout for Gurobi (leave some buffer for process overhead)
        gurobi_timeout = max(1, timeout - 2)  # Leave 2 seconds buffer
        result = execute_gurobi(script_content, timeout=gurobi_timeout)
        queue.put(result)
    except Exception as e:
        queue.put({"success": False, "value": str(e)})


def extract_code_from_output(output: str) -> str:
    """Extract Python code from model output."""
    start = output.find("```python")
    if start == -1:
        start = output.find("```")
        if start == -1:
            return None
        # Try to find code block without python marker
        end = output.find("```", start + 3)
        if end == -1:
            return None
        code = output[start+3:end].strip()
        # Check if it looks like Python code
        if "import" in code or "def " in code or "=" in code:
            return code
        return None
    end = output.find("```", start + 9)
    if end == -1:
        return None
    return output[start+9:end].strip()


def compile_script(script_content, timeout=300):
    """Execute Python script and capture results using Gurobi evaluation with timeout.
    Uses multiprocessing to enforce timeout reliably."""
    import tempfile
    # Ensure os is available (it's imported at module level)
    # Explicitly reference it to avoid scoping issues
    global os
    target_dir = './eval_execute'
    os.makedirs(target_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.py', dir=target_dir) as tmp_file:
        tmp_file_name = tmp_file.name
        tmp_file.write(script_content.encode())

    try:
        # Use multiprocessing with timeout to actually enforce the timeout
        # This can interrupt blocking calls like Gurobi's optimize()
        import multiprocessing
        import time
        
        # Use 'spawn' start method for better timeout enforcement on Linux
        # 'spawn' creates a fresh Python interpreter, which is more reliable for killing
        try:
            ctx = multiprocessing.get_context('spawn')
        except RuntimeError:
            # Context already set, use default context
            ctx = multiprocessing
        
        queue = ctx.Queue(maxsize=1)  # Small queue size
        # Use module-level function for pickling compatibility
        process = ctx.Process(target=_run_gurobi_in_process, args=(script_content, timeout, queue))
        process.daemon = False  # Don't use daemon - we want to control termination
        process.start()
        
        # Wait with strict timeout
        start_time = time.time()
        process.join(timeout=timeout)
        elapsed = time.time() - start_time
        
        if process.is_alive():
            # Process exceeded timeout, kill it forcefully
            import signal
            try:
                # Try to terminate gracefully first
                process.terminate()
                process.join(timeout=0.5)
            except:
                pass
                
            if process.is_alive():
                try:
                    # Force kill the process
                    process.kill()
                    process.join(timeout=0.1)
                except:
                    pass
                    
                # If still alive, try to kill by PID (last resort)
                if process.is_alive() and process.pid:
                    try:
                        os.kill(process.pid, signal.SIGKILL)
                        process.join(timeout=0.1)
                    except:
                        pass
            
            execution_result = f"Code execution timed out after {timeout} seconds (elapsed: {elapsed:.1f}s)"
            execution_best_solution = None
            execution_state = "Execution Failed: Timeout"
        else:
            # Process completed, get result
            try:
                # Use timeout on queue.get to avoid hanging
                result = queue.get(timeout=2)
                if result["success"]:
                    execution_best_solution = str(result["value"])
                    execution_state = "Execution Successful and Best Solution Found"
                    execution_result = f"Optimal value: {result['value']}"
                else:
                    execution_best_solution = None
                    execution_state = f"Execution Failed: {result['value']}"
                    execution_result = result["value"]
            except:
                execution_result = "Code execution failed (could not get result from queue)"
                execution_best_solution = None
                execution_state = "Execution Failed: Queue timeout"
                
    except Exception as e:
        execution_result = f"Execution error: {str(e)}"
        execution_best_solution = None
        execution_state = f"Execution Failed: {str(e)}"
    finally:
        if os.path.exists(tmp_file_name):
            os.remove(tmp_file_name)

    return {
        "execution_result": execution_result,
        "execution_best_solution": execution_best_solution, 
        "execution_state": execution_state
    }


def assess_code_correctness(code: str, ground_truth: Any, numerical_tolerance: float = 0.05, timeout: int = 15) -> Dict[str, Any]:
    """Assess code correctness using the same logic as eval.py."""
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
    
    # Execute the code using compile_script (which uses multiprocessing timeout)
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


class CodeGenerationEvalCallback(TrainerCallback):
    """
    Custom callback that evaluates model on code generation tasks.
    Evaluates on separate datasets (IndustryOR, MAMO easy, optmath) and reports metrics for each.
    """
    
    def __init__(
        self,
        eval_datasets: Dict[str, str],  # {dataset_name: dataset_path}
        tokenizer,
        model_name_or_path: str,  # Model path for vLLM
        max_new_tokens: int = 10000,  # Match eval.py default
        temperature: float = 0.0,
        numerical_tolerance: float = 0.05,
        eval_steps: int = 50,
        max_eval_samples: int = 10,  # Maximum number of samples to evaluate per dataset
        skip_code_execution: bool = False,  # Set to True to skip code execution for faster evaluation
        timeout: int = 30,  # Match run_eval.sh timeout
        tensor_parallel_size: int = 1,  # vLLM tensor parallelism
        gpu_memory_utilization: float = 0.8,  # vLLM GPU memory utilization
        use_vllm: bool = True,  # Use vLLM for inference (faster) or model.generate()
    ):
        """
        Args:
            eval_datasets: Dictionary mapping dataset names to file paths
            tokenizer: Tokenizer for the model
            model_name_or_path: Model path for vLLM loading
            max_new_tokens: Maximum tokens to generate (default: 10000, matching eval.py)
            temperature: Sampling temperature (0.0 for greedy)
            numerical_tolerance: Tolerance for numerical accuracy
            eval_steps: Evaluate every N steps
            max_eval_samples: Maximum number of samples to evaluate per dataset (default: 10)
            skip_code_execution: If True, skip code execution for faster evaluation (default: False)
            timeout: Timeout for code execution in seconds (default: 30, matching run_eval.sh)
            tensor_parallel_size: vLLM tensor parallelism (default: 1)
            gpu_memory_utilization: vLLM GPU memory utilization (default: 0.8)
            use_vllm: Use vLLM for inference (default: True, much faster than model.generate())
        """
        self.eval_datasets = eval_datasets
        self.tokenizer = tokenizer
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.numerical_tolerance = numerical_tolerance
        self.eval_steps = eval_steps
        self.max_eval_samples = max_eval_samples
        self.skip_code_execution = skip_code_execution
        self.timeout = timeout
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        self.vllm_model = None  # Will be initialized lazily
        
        # Load evaluation datasets (sample max_eval_samples examples per dataset for faster evaluation)
        self.dataset_data = {}
        logger.info(f"CodeGenerationEvalCallback initialized: max_eval_samples={self.max_eval_samples}, max_new_tokens={self.max_new_tokens}, skip_code_execution={self.skip_code_execution}")
        for name, path in eval_datasets.items():
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    all_examples = json.load(f)
                    # Sample max_eval_samples examples for faster evaluation during training
                    import random
                    random.seed(42)  # Fixed seed for reproducibility
                    if len(all_examples) > self.max_eval_samples:
                        self.dataset_data[name] = random.sample(all_examples, self.max_eval_samples)
                        logger.info(f"Loaded {self.max_eval_samples} samples (from {len(all_examples)} total) from {name}")
                    else:
                        self.dataset_data[name] = all_examples
                        logger.info(f"Loaded {len(self.dataset_data[name])} examples from {name}")
            else:
                logger.warning(f"Evaluation dataset {path} not found, skipping {name}")
    
    def _generate_prompt(self, example: Dict[str, Any]) -> str:
        """Generate prompt from example using the template from eval.py."""
        # Extract question/description
        if "description" in example:
            question = example["description"]
        elif "question" in example:
            question = example["question"]
        elif "messages" in example:
            # Extract from messages format
            for msg in example["messages"]:
                if msg["role"] == "user":
                    question = msg["content"]
                    # Remove the "Please solve..." prefix if present
                    if "Please solve this optimization problem:" in question:
                        question = question.split("Please solve this optimization problem:")[-1].strip()
                    break
        else:
            raise ValueError(f"Unknown example format: {example.keys()}")
        
        return TEMPLATE_q2mc_en.replace("{Question}", question.strip()).strip()
    
    def _get_vllm_model(self, checkpoint_path: Optional[str] = None):
        """Get or initialize vLLM model. Uses checkpoint if available, otherwise original model path."""
        if not self.use_vllm:
            return None
        
        # Use checkpoint path if available (for evaluating trained model), otherwise use original model path
        model_path = checkpoint_path if checkpoint_path and os.path.exists(checkpoint_path) else self.model_name_or_path
        
        # Initialize vLLM model if not already done or if path changed
        if self.vllm_model is None or (hasattr(self, '_last_model_path') and self._last_model_path != model_path):
            logger.info(f"Initializing vLLM model from {model_path}")
            try:
                self.vllm_model = LLM(
                    model=model_path,
                    tensor_parallel_size=self.tensor_parallel_size,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    trust_remote_code=True
                )
                self._last_model_path = model_path
                logger.info("vLLM model initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize vLLM model: {e}. Falling back to model.generate()")
                self.use_vllm = False
                return None
        
        return self.vllm_model
    
    def _evaluate_dataset(self, model, dataset_name: str, examples: List[Dict[str, Any]], state: Optional[TrainerState] = None) -> Dict[str, float]:
        """Evaluate model on a single dataset using vLLM for fast inference."""
        total_samples = 0
        code_extraction_success = 0
        execution_success = 0
        mathematical_accuracy = 0
        
        # Only evaluate on rank 0 to avoid duplicate work
        try:
            import torch.distributed as dist
            if dist.is_initialized() and dist.get_rank() != 0:
                # Return empty metrics for non-rank-0 processes
                return {
                    f"{dataset_name}_total_samples": 0,
                    f"{dataset_name}_code_extraction_rate": 0.0,
                    f"{dataset_name}_execution_success_rate": 0.0,
                    f"{dataset_name}_mathematical_accuracy": 0.0,
                }
        except (ImportError, AttributeError):
            # Distributed training not available, proceed normally
            pass
        
        # Use vLLM for fast inference (required, no fallback)
        if not self.use_vllm:
            logger.error(f"vLLM is required for evaluation but not available. Skipping {dataset_name} evaluation.")
            return {
                f"{dataset_name}_total_samples": 0,
                f"{dataset_name}_code_extraction_rate": 0.0,
                f"{dataset_name}_execution_success_rate": 0.0,
                f"{dataset_name}_mathematical_accuracy": 0.0,
            }
        
        # Get checkpoint path if available
        checkpoint_path = None
        if state and hasattr(state, 'output_dir') and state.output_dir:
            # Check for latest checkpoint
            checkpoint_path = state.output_dir
        
        vllm_model = self._get_vllm_model(checkpoint_path)
        if not vllm_model:
            logger.error(f"Failed to initialize vLLM model. Skipping {dataset_name} evaluation.")
            return {
                f"{dataset_name}_total_samples": 0,
                f"{dataset_name}_code_extraction_rate": 0.0,
                f"{dataset_name}_execution_success_rate": 0.0,
                f"{dataset_name}_mathematical_accuracy": 0.0,
            }
        
        # Use vLLM for fast batch inference (same as eval.py)
        logger.info(f"Using vLLM for evaluation of {dataset_name}")
        prompts = [self._generate_prompt(example) for example in examples]
        
        # Create sampling params (same as eval.py)
        stop_tokens = ["</s>"]
        sampling_params = SamplingParams(
            n=1,
            temperature=self.temperature,
            top_p=1.0 if self.temperature == 0 else 0.95,
            max_tokens=self.max_new_tokens,
            stop=stop_tokens
        )
        
        # Generate all prompts at once (vLLM handles batching efficiently)
        generations = vllm_model.generate(prompts, sampling_params)
        
        # Process results
        for example, generation in zip(examples, generations):
            try:
                outputs = generation.outputs
                if not outputs:
                    generated_text = ""
                else:
                    generated_text = outputs[0].text
                
                # Extract code
                extracted_code = extract_code_from_output(generated_text)
                
                # Get ground truth
                ground_truth = example.get("ground_truth", example.get("answer"))
                
                # Assess correctness
                if extracted_code:
                    code_extraction_success += 1
                    if self.skip_code_execution:
                        # Skip code execution for faster evaluation
                        correctness_metrics = {
                            "execution_success": False,
                            "mathematical_accuracy": False
                        }
                        execution_output = {
                            "execution_result": "Skipped (skip_code_execution=True)",
                            "execution_best_solution": None,
                            "execution_state": "Skipped"
                        }
                    else:
                        # Use same timeout as run_eval.sh (30 seconds)
                        correctness_metrics, execution_output = assess_code_correctness(
                            extracted_code, ground_truth, self.numerical_tolerance, timeout=self.timeout
                        )
                        if correctness_metrics["execution_success"]:
                            execution_success += 1
                        if correctness_metrics["mathematical_accuracy"]:
                            mathematical_accuracy += 1
            except Exception as e:
                logger.warning(f"Error evaluating example in {dataset_name}: {e}")
            
            total_samples += 1
        
        # Calculate metrics
        metrics = {
            f"{dataset_name}_total_samples": total_samples,
            f"{dataset_name}_code_extraction_rate": code_extraction_success / total_samples if total_samples > 0 else 0.0,
            f"{dataset_name}_execution_success_rate": execution_success / total_samples if total_samples > 0 else 0.0,
            f"{dataset_name}_mathematical_accuracy": mathematical_accuracy / total_samples if total_samples > 0 else 0.0,
        }
        
        return metrics
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, model=None, logs=None, **kwargs):
        """Run evaluation at specified steps."""
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            try:
                import torch.distributed as dist
                is_main_process = not dist.is_initialized() or dist.get_rank() == 0
            except (ImportError, AttributeError):
                is_main_process = True
            
            if is_main_process:
                logger.info(f"Running code generation evaluation at step {state.global_step}")
            
            # Evaluate on each dataset separately
            all_metrics = {}
            # Add progress bar for overall evaluation
            dataset_names = list(self.dataset_data.keys())
            for dataset_name in tqdm(dataset_names, desc="Evaluating datasets", disable=not is_main_process):
                examples = self.dataset_data[dataset_name]
                try:
                    metrics = self._evaluate_dataset(model, dataset_name, examples, state)
                    all_metrics.update(metrics)
                    if is_main_process:
                        logger.info(f"{dataset_name} - Execution Success: {metrics[f'{dataset_name}_execution_success_rate']:.3f}, "
                                  f"Mathematical Accuracy: {metrics[f'{dataset_name}_mathematical_accuracy']:.3f}")
                except Exception as e:
                    logger.error(f"Error evaluating {dataset_name}: {e}")
                    all_metrics.update({
                        f"{dataset_name}_total_samples": 0,
                        f"{dataset_name}_code_extraction_rate": 0.0,
                        f"{dataset_name}_execution_success_rate": 0.0,
                        f"{dataset_name}_mathematical_accuracy": 0.0,
                    })
            
            # Log metrics (only on main process)
            if is_main_process:
                if logs is not None:
                    logs.update(all_metrics)
                
                # Also log to wandb if available
                if hasattr(args, 'report_to') and args.report_to and 'wandb' in args.report_to:
                    try:
                        import wandb
                        wandb.log(all_metrics, step=state.global_step)
                    except ImportError:
                        pass

