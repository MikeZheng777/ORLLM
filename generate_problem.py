

import argparse
import json
import os
import re
from vllm import LLM, SamplingParams

MATH_PROBLEM_PROMPT = r"""
%=== General Mixed-Integer Linear Program (MILP) ===
% Dimensions (for reference):
% x \in R^{n_c}  (continuous vars),  y \in Z^{n_i} (integer vars)
% A \in R^{m_\le \times n_c}, B \in R^{m_\le \times n_i}, b \in R^{m_\le}
% E \in R^{m_= \times n_c}, F \in R^{m_= \times n_i}, h \in R^{m_=}
% \ell_x,u_x \in (\mathbb{R}\cup\{\pm\infty\})^{n_c}, \ \ell_y,u_y \in (\mathbb{Z}\cup\{\pm\infty\})^{n_i}

\begin{align}
\min_{x \in \mathbb{R}^{n_c},\, y \in \mathbb{Z}^{n_i}} \quad
    & c^\top x + d^\top y \label{milp:obj}\\
\text{s.t.} \quad
    & A x + B y \le b, \label{milp:ineq}\\
    & E x + F y = h, \label{milp:eq}\\
    & \ell_x \le x \le u_x, \label{milp:x-bounds}\\
    & \ell_y \le y \le u_y. \label{milp:y-bounds}
\end{align}

% Notes:
% - Replace "min" by "max" for maximization problems.
% - If y are binaries, use y \in \{0,1\}^{n_i} instead of integer bounds in \eqref{milp:y-bounds}.
"""

applications = {
    "Facility location": "Choose sites and capacities to meet demand at minimum cost.",
    # "Vehicle routing": "Plan vehicle routes and schedules with capacities.",
    # "Production planning & lot sizing": "Decide what to make, when, and in what batch sizes under setup/capacity limits.",
    # "Job-shop/flow-shop scheduling": "Sequence jobs on machines to minimize makespan or tardiness.",
    # "Inventory optimization": "Set order quantities and timing to balance holding, shortage, and setup costs.",
    # "Workforce rostering": "Assign shifts to staff to satisfy coverage, skills, and fairness constraints.",
    # "Unit commitment": "Switch generators on/off and dispatch to meet load with ramping/reserve limits.",
    # "Network design": "Select links and capacities to route flows within reliability and budget limits.",
    # "Portfolio optimization": "Choose assets and lot sizes with cardinality/turnover limits to maximize risk-adjusted return.",
    # "Cutting stock/bin packing": "Cut raw materials into orders to minimize trim loss and setups.",
}

# === Problem Generation Utilities ===

INSTRUCTION_PROMPT_TEMPLATE = (
    "You are to write a realistic, self-contained natural language problem description from a domain expert's perspective.\n"
    "The problem should be written as if coming from a business manager, operations lead, or domain expert who needs to solve a real-world decision-making problem.\n"
    "The MILP schema below is for your reference only to ensure the problem structure is appropriate (do not mention MILP, optimization, or mathematical modeling in the output).\n\n"
    "MILP schema (for your reference only, do not mention this in the output):\n"
    f"{MATH_PROBLEM_PROMPT}\n\n"
    "Application domain: {application_name} — {application_brief}.\n\n"
    "Requirements for the problem description:\n"
    "- Write from the perspective of a non-expert in optimization who needs to make a decision (e.g., a manager, operations lead, or domain expert).\n"
    "- Provide a realistic real-world scenario with concrete, plausible DATA (numbers, units).\n"
    "- Include all relevant inputs needed (e.g., costs, capacities, demands, times, setup costs, bounds, limits).\n"
    "- Include a clear objective stated in business terms (e.g., minimize cost, maximize profit, minimize time) without using optimization jargon.\n"
    "- Describe all constraints and requirements in plain language (e.g., demand must be met, capacity limits, scheduling requirements, resource availability).\n"
    "- Scale: 8–25 entities (e.g., products/customers/machines/routes) with heterogeneous parameters.\n"
    "- Keep it concise but complete. Target length: 200-500 words (approximately 1-3 paragraphs). Avoid excessive detail.\n"
    "- DO NOT mention: MILP, optimization, mathematical modeling, linear programming, formulation, decision variables, constraints (as an optimization term), objective function, or any optimization terminology.\n"
    "- DO NOT ask to 'formulate as a MILP' or 'model as an optimization problem'.\n"
    "- Output MUST be ONLY the problem statement in natural language from a domain expert's perspective. No solution, no code, no equations, no optimization terminology.\n"
    "- Output ONLY the problem description itself - nothing else. No explanations, no summaries, no repetitions, no meta-commentary.\n"
    "- Stop immediately after completing the problem description. Do not add any additional commentary, summary, explanation, or meta-commentary.\n"
    "- When you finish describing the problem, STOP. \n"
    "- Do not repeat the problem description. Output it only once.\n"
    "- The output should end with the last sentence of the actual problem description, nothing more.\n"
    "- If you include data tables or listings, make sure they are complete. Do not output incomplete or truncated data.\n\n"
)

def build_problem_prompt(application_name: str) -> str:
    brief = applications.get(application_name, "")
    # Use .replace() instead of .format() to avoid issues with LaTeX braces in MATH_PROBLEM_PROMPT
    prompt = INSTRUCTION_PROMPT_TEMPLATE.replace("{application_name}", application_name)
    prompt = prompt.replace("{application_brief}", brief)
    return prompt

def clean_generated_problem(text: str) -> str:
    """
    Clean and extract only the problem description from generated text.
    Removes extra commentary, explanations, or unwanted content.
    """
    cleaned = text.strip()
    
    # First, remove prompt template leakage (the prompt itself appearing in output)
    # Look for markers that indicate the prompt template leaked through
    prompt_markers = [
        "You are to write a realistic",
        "MILP schema",
        "Application domain:",
        "Requirements for the problem description:",
        "Write from the perspective",
        "Provide a realistic real-world scenario",
        "\\begin{align}",
        "\\min_{x",
    ]
    for marker in prompt_markers:
        idx = cleaned.find(marker)
        if idx != -1:
            cleaned = cleaned[:idx].strip()
            break
    
    # Remove "STOP" markers and any repetitive STOP patterns
    # Pattern: "STOP" followed by newlines, possibly repeated many times
    cleaned = re.sub(r"\s*STOP\s*(STOP\s*)*", "", cleaned, flags=re.IGNORECASE)
    # Also remove "(End of problem description)" or similar markers
    cleaned = re.sub(r"\(End of problem description\)\s*", "", cleaned, flags=re.IGNORECASE)
    
    # Remove common unwanted patterns
    unwanted_patterns = [
        r"\n\nNote:.*",
        r"\n\nIn summary.*",
        r"\n\nThis problem.*",
        r"\n\nTo solve.*",
        r"\n\nThe solution.*",
        r"\n\nAdditional.*",
        r"\n\n---.*",
        r"\n\n===.*",
        r"\n\n###.*",
        r"\n\n##.*",
        r"\n\n#.*",
        r"\n\nNote that.*",
        r"\n\nIt is important.*",
        r"\n\nPlease note.*",
        r"\n\nTo summarize.*",
        r"\n\nSummary:.*",
        r"\n\nIn conclusion.*",
        r"\n\nTherefore.*",
        r"\n\nThus.*",
        r"\n\nIn this problem.*",
        r"\n\nThe objective.*",
        r"\n\nThe goal.*",
        r"\n\nWe need to.*",
        r"\n\nThis scenario.*",
    ]
    
    # Remove unwanted patterns
    for pattern in unwanted_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # Stop at common ending phrases that indicate the problem is done
    stop_phrases = [
        "\n\nNote:",
        "\n\nIn summary",
        "\n\nTo summarize",
        "\n\nSummary:",
        "\n\nThis problem",
        "\n\nTo solve",
        "\n\nThe solution",
        "\n\nIn conclusion",
        "\n\nTherefore",
        "\n\nThus",
        "\n\nAdditional",
        "\n\n---",
        "\n\n===",
        "\n\n###",
        "\n\n##",
        "\n\n# Question:",
        "\n\n# Solution:",
        "\n\n# Answer:",
        "\n\n# Problem:",
        "\n\n# Note:",
        "\n\nSTOP",
        "\n\n(End of problem description)
    ]
    
    for phrase in stop_phrases:
        idx = cleaned.find(phrase)
        if idx != -1:
            cleaned = cleaned[:idx].strip()
            break
    
    # Detect and remove repetitive constraint statements
    # Look for patterns like "We must ensure that..." repeated many times
    # Split by sentences and look for repetitive patterns
    sentences = cleaned.split('.')
    if len(sentences) > 5:
        # Check if the last many sentences are repetitive constraint statements
        # Look for patterns like "We must", "We need", "We cannot", etc. repeated
        constraint_patterns = [
            "we must ensure that",
            "we must not",
            "we must",
            "we need to",
            "we cannot",
            "we are not allowed",
            "we are required",
        ]
        
        # Count how many of the last sentences match constraint patterns
        last_sentences = sentences[-10:]  # Check last 10 sentences
        constraint_count = 0
        for sent in last_sentences:
            sent_lower = sent.strip().lower()
            if any(pattern in sent_lower for pattern in constraint_patterns):
                constraint_count += 1
        
        # If more than 6 of the last 10 sentences are constraints, likely repetitive
        if constraint_count >= 6:
            # Find where the repetitive constraints start
            # Look for the first occurrence of many constraint statements in a row
            consecutive_constraints = 0
            cut_idx = len(sentences)
            for i in range(len(sentences) - 1, -1, -1):
                sent_lower = sentences[i].strip().lower()
                if any(pattern in sent_lower for pattern in constraint_patterns):
                    consecutive_constraints += 1
                    if consecutive_constraints >= 5:  # Found 5+ consecutive constraints
                        cut_idx = i
                        break
                else:
                    consecutive_constraints = 0
            
            # If we found a cut point, remove everything from there
            if cut_idx < len(sentences):
                cleaned = '. '.join(sentences[:cut_idx]).strip()
                if cleaned and not cleaned.endswith('.'):
                    # Find the last complete sentence
                    cleaned = cleaned.rstrip('.')
                    # Add back the last sentence if it's complete
                    if cut_idx > 0:
                        last_good = sentences[cut_idx - 1].strip()
                        if last_good and not any(pattern in last_good.lower() for pattern in constraint_patterns):
                            cleaned = '. '.join(sentences[:cut_idx]).strip()
                            if cleaned and not cleaned.endswith('.'):
                                cleaned += '.'
    
    # Remove trailing explanations or summaries
    # Look for patterns like "The problem is..." at the end (likely a summary)
    lines = cleaned.split('\n')
    if len(lines) > 1:
        # Check if last few lines look like a summary/explanation
        last_lines = '\n'.join(lines[-3:]).lower()
        if any(phrase in last_lines for phrase in [
            "in summary", "to summarize", "the problem is", "this is",
            "the objective is", "the goal is", "we need", "in conclusion",
            "therefore", "thus", "this scenario", "the scenario"
        ]):
            # If the last lines seem like a summary, remove them
            # But keep the main problem description
            cleaned = '\n'.join(lines[:-3]).strip()
    
    # Also check for sentences that start with meta-commentary at the end
    # Look for patterns like "This is a problem where..." at the end
    sentences = cleaned.split('.')
    if len(sentences) > 1:
        # Check last few sentences for meta-commentary
        last_sentences = '. '.join(sentences[-2:]).lower()
        if any(phrase in last_sentences for phrase in [
            "this is a problem", "this problem", "the problem", "the objective",
            "the goal", "we need to", "this scenario", "in this problem"
        ]):
            # If they look like meta-commentary (not part of the actual problem),
            # remove them but be careful not to remove legitimate problem text
            # Only remove if it's clearly a summary/explanation
            if any(phrase in last_sentences for phrase in [
                "in summary", "to summarize", "in conclusion", "therefore"
            ]):
                cleaned = '. '.join(sentences[:-2]).strip()
                if cleaned and not cleaned.endswith('.'):
                    cleaned += '.'
    
    # Detect and remove repeated problem descriptions
    # If the same text appears twice (with some variation), keep only the first occurrence
    # Split by paragraphs and check for repetition
    paragraphs = cleaned.split('\n\n')
    if len(paragraphs) > 3:
        # Check if the last few paragraphs are a repetition of earlier ones
        # Simple heuristic: if last paragraph is very similar to an earlier one, remove it
        last_para = paragraphs[-1].strip().lower()
        if len(last_para) > 50:  # Only check substantial paragraphs
            # Check if any earlier paragraph is very similar (90%+ word overlap)
            found_repetition = False
            for i in range(len(paragraphs) - 2, max(0, len(paragraphs) - 10), -1):
                earlier_para = paragraphs[i].strip().lower()
                if len(earlier_para) > 50:
                    # Simple similarity check: word overlap
                    last_words = set(last_para.split())
                    earlier_words = set(earlier_para.split())
                    if len(last_words) > 0 and len(earlier_words) > 0:
                        overlap = len(last_words & earlier_words) / len(last_words)
                        if overlap > 0.7:  # 70%+ word overlap suggests repetition
                            # Remove the repeated paragraph and everything after
                            cleaned = '\n\n'.join(paragraphs[:i+1]).strip()
                            found_repetition = True
                            break
    
    # Remove incomplete data listings (truncated tables/lists)
    # Look for patterns like "Store 5: 1." or "From Site 7 to: A=" - incomplete data entries
    # These often appear at the end and are clearly truncated
    lines = cleaned.split('\n')
    if len(lines) > 1:
        # Check last few lines for incomplete data patterns
        incomplete_patterns = [
            r'^[A-Z][a-z]+ \d+:\s*\d+\.?\s*$',  # "Store 5: 1."
            r'^[A-Z][a-z]+ \d+ to:\s*[A-Z]=\s*$',  # "From Site 7 to: A="
            r'^[A-Z][a-z]+ \d+:\s*[A-Z]=\s*$',  # "Site 1: A="
            r':\s*[A-Z]=\s*$',  # "to: A=" (incomplete)
            r':\s*\d+\.\s*$',  # ": 1." (incomplete)
        ]
        
        # Remove incomplete lines at the end
        removed_any = True
        while removed_any and len(lines) > 1:
            removed_any = False
            last_line = lines[-1].strip()
            if last_line:
                # Check if last line matches incomplete pattern
                for pattern in incomplete_patterns:
                    if re.match(pattern, last_line, re.IGNORECASE):
                        lines = lines[:-1]
                        removed_any = True
                        break
                # Also remove if last line is very short and looks incomplete
                if not removed_any and len(last_line) < 15 and (':' in last_line or '=' in last_line):
                    # Check if it's a data entry (starts with capital or number)
                    if re.match(r'^[A-Z0-9]', last_line):
                        lines = lines[:-1]
                        removed_any = True
        
        cleaned = '\n'.join(lines).strip()
    
    # Final cleanup: remove any trailing incomplete sentences (likely truncated)
    # If the last sentence is very short and doesn't end properly, remove it
    sentences = cleaned.split('.')
    if len(sentences) > 1:
        last_sent = sentences[-1].strip()
        # If last sentence is very short (< 20 chars) and doesn't look complete, remove it
        if len(last_sent) < 20 and last_sent:
            # Check if it looks like a fragment (starts with lowercase, has no verb, etc.)
            if last_sent[0].islower() or len(last_sent.split()) < 3:
                cleaned = '. '.join(sentences[:-1]).strip()
                if cleaned and not cleaned.endswith('.'):
                    cleaned += '.'
            # Also check for incomplete data entries (like "Store 5: 1" without period)
            elif ':' in last_sent and len(last_sent.split()) < 4:
                # Likely an incomplete data entry
                cleaned = '. '.join(sentences[:-1]).strip()
                if cleaned and not cleaned.endswith('.'):
                    cleaned += '.'
    
    # Truncate overly long outputs (more than ~800 words is likely too verbose)
    word_count = len(cleaned.split())
    if word_count > 800:
        # Find a good truncation point - try to cut at a sentence boundary before 600 words
        words = cleaned.split()
        target_words = 600
        if len(words) > target_words:
            # Find the last complete sentence before the target word count
            truncated_text = ' '.join(words[:target_words])
            # Find the last sentence boundary
            last_period = truncated_text.rfind('.')
            if last_period > len(truncated_text) * 0.5:  # Only use if we found a period in the second half
                cleaned = truncated_text[:last_period + 1].strip()
            else:
                # Fall back to paragraph boundary
                last_newline = truncated_text.rfind('\n\n')
                if last_newline > len(truncated_text) * 0.5:
                    cleaned = truncated_text[:last_newline].strip()
                else:
                    # Just truncate at word boundary
                    cleaned = truncated_text.strip()
    
    # Remove any remaining meta-commentary or explanatory text
    # Look for phrases that indicate the output is explaining what it's doing rather than describing the problem
    meta_phrases = [
        "all numbers are representative",
        "based on real data",
        "from our current operations",
        "from market analysis",
        "the data for",
        "the table below lists",
        "the following table",
        "as shown in the table",
        "see the table",
        "refer to the table",
    ]
    
    sentences = cleaned.split('.')
    for i in range(len(sentences) - 1, max(0, len(sentences) - 5), -1):
        sent_lower = sentences[i].strip().lower()
        if any(phrase in sent_lower for phrase in meta_phrases):
            # Remove this sentence and everything after
            cleaned = '. '.join(sentences[:i]).strip()
            if cleaned and not cleaned.endswith('.'):
                cleaned += '.'
            break
    
    return cleaned.strip()

def generate_problem_with_llm(model: LLM, application_name: str, max_tokens: int = 1024,
                              temperature: float = 0.7, top_p: float = 0.95) -> str:
    """Legacy function for single generation (kept for backward compatibility)."""
    prompt = build_problem_prompt(application_name)
    
    # Enhanced stop tokens to catch when generation should stop
    stop_tokens = [
        "</s>",
        "\n\nNote:",
        "\n\nIn summary",
        "\n\nTo summarize",
        "\n\nSummary:",
        "\n\nThis problem",
        "\n\nTo solve",
        "\n\nThe solution",
        "\n\nIn conclusion",
        "\n\nTherefore",
        "\n\nThus",
        "\n\nAdditional",
        "\n\n---",
        "\n\n===",
        "\n\n###",
        "\n\n##",
        "\n\n# Question:",
        "\n\n# Solution:",
        "\n\n# Answer:",
        "\n\n# Problem:",
        "\n\n# Note:",
        "\n\nSTOP",
        "\nSTOP",
        "(End of problem description)",
    ]
    
    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop_tokens
    )
    outputs = model.generate([prompt], sampling_params)
    raw_output = outputs[0].outputs[0].text.strip()
    
    # Clean the output to remove unwanted content
    cleaned_output = clean_generated_problem(raw_output)
    return cleaned_output

def save_generated_problems(problems, output_json_path: str) -> None:
    """Save a list of generated problems to a .json file in the requested schema."""
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(problems, f, ensure_ascii=False, indent=2)

def generate_problems_dataset(model: LLM, selected_applications, num_per_application: int,
                              max_tokens: int, temperature: float, top_p: float, 
                              use_batching: bool = True, pre_tokenize: bool = False) -> list:
    """
    Generate problems dataset using batched generation for better performance.
    
    Args:
        model: vLLM model instance
        selected_applications: List of application names
        num_per_application: Number of problems per application
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p nucleus sampling
        use_batching: If True, batch all prompts in one call (faster, uses prefix caching)
        pre_tokenize: If True, pre-tokenize prompts to reduce CPU overhead
    """
    dataset = []
    problem_counter = 0
    
    if use_batching:
        # Option A: Batch all prompts in one call - leverages prefix caching automatically
        # Build all prompts upfront
        prompts = []
        app_names_expanded = []
        for app_name in selected_applications:
            for _ in range(num_per_application):
                prompt = build_problem_prompt(app_name)
                prompts.append(prompt)
                app_names_expanded.append(app_name)
        
        # Pre-tokenize if requested (reduces tokenization overhead on GPU)
        if pre_tokenize:
            # Get tokenizer from model
            tokenizer = model.llm_engine.tokenizer.tokenizer
            prompt_token_ids = []
            for prompt in prompts:
                token_ids = tokenizer.encode(prompt, add_special_tokens=False)
                prompt_token_ids.append(token_ids)
            # Use token IDs directly (vLLM supports this via prompt_adapter)
            # Note: vLLM's generate() accepts prompts as strings, but we can optimize by
            # using the tokenizer cache. For now, we'll pass prompts and let vLLM handle it.
            # The main benefit of batching is already achieved above.
            pass
        
        # Single batched generation call - vLLM automatically uses prefix caching
        # for shared prefixes across prompts
        # Enhanced stop tokens to catch when generation should stop
        # stop_tokens = [
        #     "</s>",
        #     "\n\nNote:",
        #     "\n\nIn summary",
        #     "\n\nTo summarize",
        #     "\n\nSummary:",
        #     "\n\nThis problem",
        #     "\n\nTo solve",
        #     "\n\nThe solution",
        #     "\n\nIn conclusion",
        #     "\n\nTherefore",
        #     "\n\nThus",
        #     "\n\nAdditional",
        #     "\n\n---",
        #     "\n\n===",
        #     "\n\n###",
        #     "\n\n##",
        #     "\n\n# Question:",
        #     "\n\n# Solution:",
        #     "\n\n# Answer:",
        #     "\n\n# Problem:",
        #     "\n\n# Note:",
        #     "\n\nSTOP",
        #     "\nSTOP",
        #     "(End of problem description)",
        # # ]
        
        # Enhanced stop tokens to catch when generation should stop
        stop_tokens = [
            "</s>",
            "\n\nNote:",
            "\n\nIn summary",
            "\n\nTo summarize",
            "\n\nSummary:",
            "\n\nThis problem",
            "\n\nTo solve",
            "\n\nThe solution",
            "\n\nIn conclusion",
            "\n\nTherefore",
            "\n\nThus",
            "\n\nAdditional",
            "\n\n---",
            "\n\n===",
            "\n\n###",
            "\n\n##",
            "\n\n# Question:",
            "\n\n# Solution:",
            "\n\n# Answer:",
            "\n\n# Problem:",
            "\n\n# Note:",
            "\n\nSTOP",
            "\nSTOP",
            "(End of problem description)",
        ]
        
        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            # stop=stop_tokens
        )
        
        print(f"Generating {len(prompts)} problems in a single batched call...")
        outputs = model.generate(prompts, sampling_params)
        
        # Process results and clean them
        for i, output in enumerate(outputs):
            app_name = app_names_expanded[i]
            raw_description = output.outputs[0].text.strip()
            # Clean the output to remove unwanted content
            # description = clean_generated_problem(raw_description)
            description = raw_description   
            dataset.append({
                "problem_id": problem_counter,
                "application": app_name,
                "description": description,
                "ground_truth": None,
                "correct_program": None,
            })
            problem_counter += 1
    else:
        # Legacy: one-by-one generation (slower but simpler)
        for app_name in selected_applications:
            for _ in range(num_per_application):
                description = generate_problem_with_llm(
                    model,
                    app_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                dataset.append({
                    "problem_id": problem_counter,
                    "application": app_name,
                    "description": description,
                    "ground_truth": None,
                    "correct_program": None,
                })
                problem_counter += 1
    
    return dataset


    

def parse_args():
    parser = argparse.ArgumentParser(description="Generate realistic NL MILP problems for given applications")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or HF ID of the model")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="GPU memory utilization ratio")
    parser.add_argument("--enable_triton_moe", action="store_true", default=False,
                       help="Enable Triton MoE backend for better MoE performance")
    parser.add_argument("--max_model_len", type=int, default=None,
                       help="Maximum sequence length (auto-detect if None)")

    # Generation behavior
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p nucleus sampling")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens to generate")

    # Problem generation controls
    parser.add_argument("--problem_gen_output", type=str, default=None,
                       help="If set, generate MILP problems and save to this JSON path, then exit")
    parser.add_argument("--problem_gen_apps", type=str, default="all",
                       help="Comma-separated application names or 'all' to use all predefined applications")
    parser.add_argument("--problem_gen_num_per_app", type=int, default=10,
                       help="How many problems to generate per selected application")
    parser.add_argument("--disable_batching", action="store_true", default=False,
                       help="Disable batched generation (slower, uses legacy one-by-one method)")
    parser.add_argument("--pre_tokenize", action="store_true", default=False,
                       help="Pre-tokenize prompts to reduce CPU overhead (experimental)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # If problem generation is requested, run that path first and exit
    if args.problem_gen_output is not None:
        # Init model for generation
        # Set up MoE optimizations if requested
        if args.enable_triton_moe:
            import os
            os.environ["VLLM_USE_TRITON_MOE"] = "1"
            print("Using Triton MoE backend for better performance")
        
        # enforce_eager=True disables torch.compile and CUDA graphs to avoid MoE extension errors
        model_kwargs = {
            "model": args.model_name_or_path,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "trust_remote_code": True,
            "enforce_eager": True,
            "enable_prefix_caching": True,  # Enable prefix caching for shared prompt prefixes
        }
        if args.max_model_len is not None:
            model_kwargs["max_model_len"] = args.max_model_len
        
        model = LLM(**model_kwargs)
        if args.problem_gen_apps.strip().lower() == "all":
            selected_apps = list(applications.keys())
        else:
            requested = [s.strip() for s in args.problem_gen_apps.split(',') if s.strip()]
            # Keep only known applications to avoid accidental typos
            selected_apps = [a for a in requested if a in applications]
            if not selected_apps:
                selected_apps = list(applications.keys())
        dataset = generate_problems_dataset(
            model,
            selected_apps,
            num_per_application=args.problem_gen_num_per_app,
            max_tokens=args.max_tokens or 1024,
            temperature=args.temperature,
            top_p=args.top_p,
            use_batching=not args.disable_batching,
            pre_tokenize=args.pre_tokenize,
        )
        save_generated_problems(dataset, args.problem_gen_output)
        print(f"Generated {len(dataset)} problems across {len(selected_apps)} applications → {args.problem_gen_output}")
    else:
        print("No --problem_gen_output specified. Nothing to do.")
