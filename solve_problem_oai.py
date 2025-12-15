import argparse
import json
import os
import re
import sys
import subprocess
import tempfile
import concurrent.futures
from collections import Counter
import openai
from tqdm import tqdm
import math
import copy
from datetime import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from utils.gurobi import execute_gurobi, test_optimality
import gurobipy as gp
from gurobipy import GRB

# Load environment variables from .env file if it exists
def load_env_file():
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

TEMPLATE_q2mc_en = r"""
You are an Operations Research expert. Below is a real-world optimization problem description. Your task is to:

STEP 1: Mathematical Formulation
First, formulate the problem as a mathematical optimization model. Clearly define:
- Sets and indices
- Parameters (with their values from the problem description)
- Decision variables (with their domains: continuous, integer, binary)
- Objective function (minimize or maximize, with explicit mathematical expression)
- Constraints (all constraints with explicit mathematical expressions)

Write the mathematical model in standard OR notation using LaTeX-style math or clear mathematical expressions.

STEP 2: Gurobi Implementation
Based on your mathematical formulation, write complete Python code using 'gurobipy' to solve the optimization problem.

IMPORTANT: Your code should follow these requirements:
1. Create a Gurobi model variable named 'model' (not 'm' or other names)
2. Include all necessary imports and data setup
3. Extract all parameter values from the problem description
4. Write the complete optimization code that can run independently
5. Do NOT wrap everything in a function - write the code directly
6. Make sure the model is created and optimized in the main execution flow
7. The code should print the optimal objective value when solved successfully
8. Map all decision variables, constraints, and objective exactly as formulated in Step 1

Example format:
```python
import gurobipy as gp
from gurobipy import GRB

# Data setup - extract all values from problem description
# Sets
A = [...]  # aircraft types
R = [...]  # routes
# Parameters
availability = {...}  # availability_a for each a in A
demand = {...}  # demand_r for each r in R
costs = {...}  # costs_ar for each (a, r) pair
# ... all other parameters

# Create model
model = gp.Model("ProblemName")

# Decision variables (matching your mathematical formulation)
x = model.addVars(A, R, vtype=GRB.INTEGER, name="allocation")

# Objective (matching your mathematical formulation)
model.setObjective(
    gp.quicksum(costs[a, r] * x[a, r] for a in A for r in R),
    sense=GRB.MINIMIZE
)

# Constraints (matching your mathematical formulation)
# Constraint 1: Availability
for a in A:
    model.addConstr(
        gp.quicksum(x[a, r] for r in R) <= availability[a],
        name=f"availability_{a}"
    )
# Constraint 2: Demand
for r in R:
    model.addConstr(
        gp.quicksum(capabilities[a, r] * x[a, r] for a in A) == demand[r],
        name=f"demand_{r}"
    )
# ... all other constraints

# Optimize
model.optimize()

# Results
if model.status == GRB.OPTIMAL:
    print(f"Optimal value: {model.objVal}")
else:
    print("No optimal solution found")
```

# Problem Description:
{Question}

# Response:
First, provide your mathematical formulation, then provide the complete Python code.
"""

ONE_QUESTION = r"""
A lab has 1000 units of medicinal ingredients to make two pills, a large pill and a small pill. A large pill requires 3 units of medicinal ingredients and 2 units of filler. A small pill requires 2 units of medicinal ingredients and 1 unit of filler. The lab has to make at least 100 large pills. However, since small pills are more popular at least 60% of the total number of pills must be small. How many of each should be made to minimize the total number of filler material needed?
"""

ADD_SCRIPT = '\nif model.status == GRB.OPTIMAL:\n    print(f"Just print the best solution: {model.objVal}")\nelse:\n    print("No Best Solution")'

def majority_voting(pred_answers):
    """Count occurrences and return the most frequent answer."""
    count = Counter(pred_answers)
    max_count = max(count.values())
    possible_answers = [answer for answer, cnt in count.items() if cnt == max_count]
    return possible_answers[0]

def extract_mathematical_formulation(output):
    """Extract mathematical formulation from model output (before code block)."""
    # Look for code block to find where formulation ends
    code_start = output.find("```python")
    if code_start == -1:
        code_start = output.find("```")
    
    if code_start == -1:
        # No code found, return everything
        return output.strip()
    
    # Extract everything before the code block
    formulation = output[:code_start].strip()
    
    # Try to find where STEP 1 or mathematical formulation starts
    step1_markers = ["STEP 1", "Mathematical Formulation", "Mathematical formulation", 
                     "## Mathematical", "Mathematical Model"]
    for marker in step1_markers:
        idx = formulation.lower().find(marker.lower())
        if idx != -1:
            formulation = formulation[idx:].strip()
            break
    
    return formulation

def extract_code_from_output(output):
    """Extract Python code from model output."""
    start = output.find("```python")
    if start == -1:
        return None
    end = output.find("```", start + 9)
    if end == -1:
        return None
    return output[start+9:end].strip()

def diagnose_infeasibility(script_content):
    """
    Diagnose infeasible Gurobi models by computing IIS and analyzing constraints.
    Returns diagnostic information including IIS constraints and variable bounds.
    IIS file is computed temporarily for analysis but not saved.
    """
    diagnostics = {
        "iis_constraints": [],
        "iis_variables": [],
        "constraint_details": [],
        "variable_details": []
    }
    
    try:
        exec_globals = {'gp': gp, 'GRB': GRB}
        exec(script_content, exec_globals)
        
        # Find the model
        model = None
        if 'model' in exec_globals:
            potential_model = exec_globals['model']
            if hasattr(potential_model, 'optimize') and hasattr(potential_model, 'status'):
                model = potential_model
        
        if model is None:
            return diagnostics
        
        # Check if model is infeasible
        if model.status == GRB.INFEASIBLE:
            try:
                model.computeIIS()
                
                # Collect IIS constraints with full details
                for constr in model.getConstrs():
                    if constr.IISConstr:
                        # Get constraint expression details
                        row = model.getRow(constr)
                        expr_parts = []
                        for i in range(row.size()):
                            var = row.getVar(i)
                            coeff = row.getCoeff(i)
                            expr_parts.append(f"{coeff}*{var.VarName}")
                        
                        diagnostics["iis_constraints"].append(constr.ConstrName)
                        diagnostics["constraint_details"].append({
                            "name": constr.ConstrName,
                            "sense": constr.Sense,
                            "rhs": constr.RHS,
                            "expression": " + ".join(expr_parts) if expr_parts else "empty"
                        })
                
                # Collect IIS variables with full details
                for var in model.getVars():
                    if var.IISLB or var.IISUB:
                        diagnostics["iis_variables"].append(var.VarName)
                        diagnostics["variable_details"].append({
                            "name": var.VarName,
                            "lower_bound_violated": var.IISLB,
                            "upper_bound_violated": var.IISUB,
                            "lb": var.LB,
                            "ub": var.UB,
                            "vtype": var.VType
                        })
                    
            except Exception as e:
                diagnostics["error"] = f"Could not compute IIS: {str(e)}"
                
    except Exception as e:
        diagnostics["error"] = f"Error diagnosing infeasibility: {str(e)}"
    
    return diagnostics

def analyze_infeasibility_pattern(diagnostics):
    """
    Analyze IIS to identify generic infeasibility patterns.
    Returns pattern type and repair strategy.
    """
    if not diagnostics.get("constraint_details"):
        return None, None
    
    constraint_details = diagnostics["constraint_details"]
    variable_details = diagnostics.get("variable_details", [])
    
    # Pattern 1: Equality constraint with upper bound constraint conflict
    equality_constraints = [c for c in constraint_details if c['sense'] in ['=', '==', 'EQUAL']]
    upper_bound_constraints = [c for c in constraint_details if c['sense'] in ['<', '<=', 'LESS_EQUAL']]
    
    if equality_constraints and upper_bound_constraints:
        # Check if they involve similar variables
        eq_vars = set()
        ub_vars = set()
        for c in equality_constraints:
            # Extract variable names from expression (simplified)
            eq_vars.update(re.findall(r'(\w+)\[', c['expression']) or re.findall(r'x\[(\w+)', c['expression']))
        for c in upper_bound_constraints:
            ub_vars.update(re.findall(r'(\w+)\[', c['expression']) or re.findall(r'x\[(\w+)', c['expression']))
        
        if eq_vars & ub_vars:  # Common variables
            return "equality_upper_bound_conflict", {
                "equality_constraints": [c['name'] for c in equality_constraints],
                "upper_bound_constraints": [c['name'] for c in upper_bound_constraints],
                "common_variables": list(eq_vars & ub_vars)
            }
    
    # Pattern 2: Multiple upper bound constraints on same variables
    if len(upper_bound_constraints) >= 2:
        # Check for overlapping variables
        constraint_vars = {}
        for c in upper_bound_constraints:
            vars_in_c = set(re.findall(r'(\w+)\[', c['expression']) or re.findall(r'x\[(\w+)', c['expression']))
            constraint_vars[c['name']] = vars_in_c
        
        # Find constraints with overlapping variables
        overlapping = []
        constraint_names = list(constraint_vars.keys())
        for i in range(len(constraint_names)):
            for j in range(i+1, len(constraint_names)):
                if constraint_vars[constraint_names[i]] & constraint_vars[constraint_names[j]]:
                    overlapping.append((constraint_names[i], constraint_names[j]))
        
        if overlapping:
            return "multiple_upper_bounds_conflict", {
                "conflicting_constraints": overlapping,
                "rhs_values": {c['name']: c['rhs'] for c in upper_bound_constraints}
            }
    
    # Pattern 3: Variable bound violations
    if variable_details:
        lb_violations = [v for v in variable_details if v['lower_bound_violated']]
        ub_violations = [v for v in variable_details if v['upper_bound_violated']]
        
        if lb_violations or ub_violations:
            return "variable_bounds_conflict", {
                "lower_bound_violations": [v['name'] for v in lb_violations],
                "upper_bound_violations": [v['name'] for v in ub_violations]
            }
    
    # Pattern 4: Equality constraint with insufficient resources
    if equality_constraints:
        # Check if RHS is large relative to available resources
        return "equality_resource_conflict", {
            "equality_constraints": [c['name'] for c in equality_constraints],
            "rhs_values": {c['name']: c['rhs'] for c in equality_constraints}
        }
    
    return "unknown_pattern", {"constraints": [c['name'] for c in constraint_details]}

def extract_parameters_from_code(code):
    """
    Extract all dictionary parameters from code generically.
    Returns a dictionary mapping (var_name, key) -> value for all dict assignments.
    """
    code_params = {}
    
    # Pattern: variable_name = {key1: value1, key2: value2, ...}
    # This matches any dictionary assignment
    dict_pattern = r'(\w+)\s*=\s*\{([^}]*)\}'
    
    for match in re.finditer(dict_pattern, code, re.MULTILINE | re.DOTALL):
        var_name = match.group(1)
        dict_content = match.group(2)
        
        # Try to parse the dictionary content
        try:
            # Use eval in a safe way to parse dictionary entries
            # Match individual key:value pairs
            entry_pattern = r'["\']?([^"\':]+)["\']?\s*:\s*(\d+\.?\d*)'
            for entry_match in re.finditer(entry_pattern, dict_content):
                key = entry_match.group(1).strip().strip("'\"")
                value = float(entry_match.group(2))
                value = int(value) if value.is_integer() else value
                
                # Store as (var_name, key) -> value
                code_params[(var_name, key)] = value
        except:
            # If parsing fails, try exec approach for the full dictionary
            try:
                exec_globals = {}
                exec(f"temp_dict = {match.group(0)}", exec_globals)
                if 'temp_dict' in exec_globals:
                    for key, value in exec_globals['temp_dict'].items():
                        code_params[(var_name, str(key))] = value
            except:
                pass
    
    return code_params

def normalize_key(key):
    """
    Normalize keys for matching (remove spaces, convert to lowercase, handle dashes).
    """
    return key.lower().replace(' ', '').replace('_', '').replace('-', '')

def find_best_match(desc_key, code_keys):
    """
    Find the best matching code key for a description key using fuzzy matching.
    """
    desc_normalized = normalize_key(desc_key)
    
    best_match = None
    best_score = 0
    
    for code_key in code_keys:
        code_normalized = normalize_key(code_key)
        
        # Simple matching: check if normalized keys are similar
        # Score based on common characters
        common_chars = sum(1 for c in desc_normalized if c in code_normalized)
        score = common_chars / max(len(desc_normalized), len(code_normalized), 1)
        
        # Bonus for exact match
        if desc_normalized == code_normalized:
            score = 1.0
        
        # Bonus for substring match
        if desc_normalized in code_normalized or code_normalized in desc_normalized:
            score = max(score, 0.8)
        
        if score > best_score:
            best_score = score
            best_match = code_key
    
    # Only return match if score is above threshold
    return best_match if best_score > 0.5 else None

def repair_code_bugs(code, problem_description, diagnostics):
    """
    Repair code bugs (parameter mismatches, variable name errors, constraint implementation bugs)
    while STRICTLY preserving the original problem constraints.
    
    This function fixes:
    - Parameter value mismatches between code and problem description
    - Variable name/index mismatches
    - Constraint implementation bugs (wrong variable used, wrong index)
    - Missing or incorrect parameter extractions
    
    It does NOT:
    - Change constraint types (== stays ==, <= stays <=)
    - Modify constraint RHS values (unless they're wrong compared to problem description)
    - Relax variable bounds (unless they're incorrectly set)
    
    Returns repaired code and list of changes made.
    """
    changes = []
    repaired_code = code
    
    if not problem_description:
        return code, ["No problem description provided for bug-fixing repair"]
    
    # Extract parameters from problem description (generic extraction)
    desc_params = extract_parameters_from_description(problem_description)
    
    # Extract parameters from code (generic extraction)
    code_params = extract_parameters_from_code(code)
    
    # Build a mapping from description keys to code (var_name, key) pairs
    # First, collect all unique keys from code
    code_keys_by_var = {}
    for (var_name, key), value in code_params.items():
        if var_name not in code_keys_by_var:
            code_keys_by_var[var_name] = []
        code_keys_by_var[var_name].append(key)
    
    # Match description parameters with code parameters
    for desc_key, desc_value in desc_params.items():
        # Try to find matching code parameter
        best_var = None
        best_code_key = None
        
        # Try exact match first
        for var_name, keys in code_keys_by_var.items():
            for code_key in keys:
                if normalize_key(desc_key) == normalize_key(code_key):
                    best_var = var_name
                    best_code_key = code_key
                    break
            if best_var:
                break
        
        # If no exact match, try fuzzy matching
        if not best_var:
            for var_name, keys in code_keys_by_var.items():
                matched_key = find_best_match(desc_key, keys)
                if matched_key:
                    best_var = var_name
                    best_code_key = matched_key
                    break
        
        # If we found a match, check for value mismatch
        if best_var and best_code_key:
            code_value = code_params.get((best_var, best_code_key))
            if code_value is not None and code_value != desc_value:
                # Fix the parameter value in code
                # Pattern: var_name = {..., 'key': old_value, ...}
                # Need to handle various quote styles and spacing
                # Try multiple patterns to handle different dictionary formats
                patterns = [
                    # Pattern 1: 'key': value or "key": value
                    rf"({re.escape(best_var)}\s*=\s*{{[^}}]*['\"]{re.escape(best_code_key)}['\"]\s*:\s*)(\d+\.?\d*)",
                    # Pattern 2: key: value (no quotes)
                    rf"({re.escape(best_var)}\s*=\s*{{[^}}]*\b{re.escape(best_code_key)}\s*:\s*)(\d+\.?\d*)",
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, repaired_code, re.MULTILINE | re.DOTALL)
                    if match:
                        old_value = match.group(2)
                        # Replace just the value part
                        new_code = match.group(1) + str(desc_value)
                        repaired_code = repaired_code.replace(match.group(0), new_code)
                        changes.append(f"Fixed parameter '{best_var}['{best_code_key}']': {old_value} → {desc_value}")
                        break
    
    # Fix constraint RHS value mismatches (generic approach)
    constraint_details = diagnostics.get("constraint_details", [])
    for constr in constraint_details:
        constr_name = constr['name']
        constr_rhs = constr['rhs']
        constr_sense = constr.get('sense', '')
        
        # Only fix equality constraints (==) where RHS should match a parameter
        if constr_sense in ['=', '==', 'EQUAL']:
            # Try to extract entity/key from constraint name
            # Common patterns: "demand_JFK-ORD", "capacity_A320", etc.
            # Extract the part after underscore or dash
            name_parts = re.split(r'[_-]', constr_name)
            if len(name_parts) >= 2:
                # Try to match with description parameters
                for desc_key, desc_value in desc_params.items():
                    # Check if constraint name contains the description key
                    if normalize_key(desc_key) in normalize_key(constr_name):
                        if constr_rhs != desc_value:
                            # Find and fix the constraint RHS in code
                            pattern = rf'(model\.addConstr\([^)]*name\s*=\s*["\']?{re.escape(constr_name)}["\']?[^)]*==\s*)(\d+\.?\d*)([^)]*\))'
                            match = re.search(pattern, repaired_code, re.MULTILINE | re.DOTALL)
                            if match:
                                old_rhs = match.group(2)
                                new_constr = match.group(1) + str(desc_value) + match.group(3)
                                repaired_code = repaired_code.replace(match.group(0), new_constr)
                                changes.append(f"Fixed constraint RHS for '{constr_name}': {old_rhs} → {desc_value}")
                                break
    
    if not changes:
        changes.append("No code bugs detected (parameters match problem description)")
    
    return repaired_code, changes

def repair_infeasible_code(code, diagnostics, problem_description):
    """
    Automatically repair infeasible Gurobi code by fixing code bugs while preserving constraints.
    
    This function fixes code bugs (parameter mismatches, constraint implementation errors)
    but does NOT modify the original problem constraints.
    
    Requires problem_description to compare code parameters against problem description.
    
    Returns repaired code and list of changes made.
    """
    if not problem_description:
        return code, ["No problem description provided - cannot perform bug-fixing repair"]
    
    return repair_code_bugs(code, problem_description, diagnostics)

def extract_parameters_from_description(problem_description):
    """
    Extract key numerical parameters from problem description text using generic patterns.
    Returns a dictionary mapping parameter identifiers to their values.
    
    Uses generic patterns to find:
    - Entity-value pairs: "Entity: value units" or "Entity - Entity: value units"
    - Dictionary-like structures: "key: value" patterns
    - Numerical values with context
    """
    params = {}
    
    # Pattern 1: Entity: value (units) - e.g., "A320: 10 frames available", "JFK-ORD: 150 passengers/day"
    # Matches: "Entity: number" or "Entity1 - Entity2: number"
    entity_value_pattern = r'([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)?)\s*[–-]?\s*([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)?)?\s*:\s*(\d+\.?\d*)\s*(?:\w+)?'
    for match in re.finditer(entity_value_pattern, problem_description, re.IGNORECASE):
        entity1 = match.group(1).strip()
        entity2 = match.group(2).strip() if match.group(2) else None
        value = float(match.group(3))
        
        if entity2:
            # Composite key like "JFK-ORD"
            key = f"{entity1}-{entity2}"
            params[key] = int(value) if value.is_integer() else value
        else:
            # Single entity
            params[entity1] = int(value) if value.is_integer() else value
    
    # Pattern 2: Slack values - "Entity: max used X, committed Y → slack Z"
    slack_pattern = r'(\w+):\s*max used \d+,\s*committed \d+\s*→\s*slack (\d+)'
    for match in re.finditer(slack_pattern, problem_description, re.IGNORECASE):
        entity = match.group(1).strip()
        slack_value = int(match.group(2))
        params[f"slack_{entity}"] = slack_value
    
    # Pattern 3: Generic "Entity has/requires/needs value" patterns
    has_pattern = r'(\w+(?:\s+\w+)?)\s+(?:has|requires|needs|must have|can have)\s+(\d+\.?\d*)'
    for match in re.finditer(has_pattern, problem_description, re.IGNORECASE):
        entity = match.group(1).strip()
        value = float(match.group(2))
        if entity not in params:  # Don't overwrite more specific patterns
            params[entity] = int(value) if value.is_integer() else value
    
    # Pattern 4: "value units per entity" - e.g., "150 passengers/day for JFK-ORD"
    per_entity_pattern = r'(\d+\.?\d*)\s+(\w+)\s+(?:per|for)\s+([A-Za-z0-9]+(?:\s*[-–]\s*[A-Za-z0-9]+)?)'
    for match in re.finditer(per_entity_pattern, problem_description, re.IGNORECASE):
        value = float(match.group(1))
        unit = match.group(2)
        entity = match.group(3).strip().replace(' ', '-')
        key = f"{entity}_{unit}"
        params[key] = int(value) if value.is_integer() else value
    
    return params

def identify_parameter_modifications(diagnostics, problem_description):
    """
    Based on IIS diagnostics, identify which parameters in the problem description
    should be modified to make the problem feasible.
    Returns a dictionary of parameter changes.
    """
    modifications = {}
    constraint_details = diagnostics.get("constraint_details", [])
    
    if not constraint_details:
        return modifications
    
    print(f"  Analyzing {len(constraint_details)} IIS constraints...")
    
    # Analyze constraint patterns to determine parameter modifications
    for constr in constraint_details:
        constr_name = constr['name']
        constr_sense = constr['sense']
        constr_rhs = constr['rhs']
        
        # Pattern: demand constraint conflicting with slot constraint
        if 'demand' in constr_name.lower() and constr_sense == '=':
            # Extract route from constraint name (e.g., "demand_JFK-ORD" -> "JFK-ORD")
            route_match = re.search(r'demand[_-]?([A-Z]+[-_][A-Z]+)', constr_name, re.IGNORECASE)
            if route_match:
                route = route_match.group(1).upper().replace('_', '-')
                route_parts = route.split('-')
                if len(route_parts) >= 2:
                    origin_airport = route_parts[0]
                    dest_airport = route_parts[1]
                    
                    # Check if there's a corresponding slot constraint in IIS
                    # Look for arrival slot at destination or departure slot at origin
                    slot_constr = None
                    for c in constraint_details:
                        if 'slot' in c['name'].lower():
                            # Check if it's for the destination airport (arrival) or origin (departure)
                            if (dest_airport in c['name'].upper() and 'arr' in c['name'].lower()) or \
                               (origin_airport in c['name'].upper() and 'dep' in c['name'].lower()):
                                slot_constr = c
                                break
                    
                    if slot_constr:
                        # Need to either reduce demand or increase slot capacity
                        # Strategy: Increase slot capacity (more conservative)
                        airport = dest_airport if 'arr' in slot_constr['name'].lower() else origin_airport
                        slot_type = 'arr' if 'arr' in slot_constr['name'].lower() else 'dep'
                        current_slack = slot_constr['rhs']
                        # Calculate minimum flights needed
                        # Assume max capacity is 180 (A320), but we need to check actual capacities
                        min_flights = math.ceil(constr_rhs / 180)  # Conservative estimate
                        if min_flights > current_slack:
                            modifications[f"{slot_type}_slack_{airport}"] = int(min_flights + 1)
        
        # Pattern: slot constraint too restrictive
        if 'slot' in constr_name.lower() and constr_sense == '<':
            # Extract airport from constraint name - handle various formats
            # Format 1: "slots_arr_ORD", "slots_dep_JFK" (plural slots first)
            # Format 2: "arr_slots_ORD", "dep_slots_JFK" (arr/dep first)
            # Format 3: "slot_arrive_ORD", "slot_depart_JFK"
            slot_type = None
            airport = None
            
            # Try format 1: slots_arr_ORD or slots_dep_JFK (plural slots first)
            match1 = re.search(r'slots?[_-](?:arr|dep)[_-](\w+)', constr_name, re.IGNORECASE)
            if match1:
                airport = match1.group(1).upper()
                slot_type = 'arr' if 'arr' in constr_name.lower() else 'dep'
            else:
                # Try format 2: arr_slots_ORD or dep_slots_JFK
                match2 = re.search(r'(?:arr|dep)[_-]slots?[_-](\w+)', constr_name, re.IGNORECASE)
                if match2:
                    airport = match2.group(1).upper()
                    slot_type = 'arr' if 'arr' in constr_name.lower() else 'dep'
                else:
                    # Try format 3: slot_arrive_ORD or slot_depart_JFK
                    match3 = re.search(r'slot[_-](?:arrive|depart)[_-](\w+)', constr_name, re.IGNORECASE)
                    if match3:
                        airport = match3.group(1).upper()
                        slot_type = 'arr' if 'arrive' in constr_name.lower() else 'dep'
                    else:
                        # Try format 4: slot_ORD (generic, extract airport from end)
                        match4 = re.search(r'slot[_-](?:arr|dep)[_-]?(\w+)$', constr_name, re.IGNORECASE)
                        if match4:
                            airport = match4.group(1).upper()
                            slot_type = 'arr' if 'arr' in constr_name.lower() else 'dep'
            
            if airport:
                if not slot_type:
                    slot_type = 'arr' if 'arr' in constr_name.lower() else 'dep'
                current_slack = constr_rhs
                # Calculate minimum flights needed based on demand if available
                # Otherwise increase slack by 50% + 1, but at least +2
                min_flights_needed = None
                for c in constraint_details:
                    if 'demand' in c['name'].lower() and c['sense'] == '=':
                        # Check if this demand constraint is related to the slot constraint
                        if airport in c['name'].upper():
                            # Estimate min flights: assume max capacity 180
                            min_flights_needed = math.ceil(c['rhs'] / 180)
                            break
                
                if min_flights_needed and min_flights_needed > current_slack:
                    new_slack = int(min_flights_needed + 1)
                else:
                    new_slack = max(int(current_slack * 1.5) + 1, int(current_slack) + 2)
                
                modifications[f"{slot_type}_slack_{airport}"] = new_slack
        
        # Pattern: demand too high relative to available resources
        if 'demand' in constr_name.lower() and constr_sense == '=':
            route_match = re.search(r'demand[_-]?([A-Z]+[-_][A-Z]+)', constr_name, re.IGNORECASE)
            if route_match:
                route = route_match.group(1).upper()
                # Reduce demand by 10% as a conservative modification
                current_demand = constr_rhs
                modifications[f"demand_{route}"] = int(current_demand * 0.9)
    
    return modifications

def modify_problem_description(problem_description, modifications):
    """
    Modify the problem description text with new parameter values.
    Returns modified problem description and list of changes made.
    """
    modified_text = problem_description
    changes = []
    
    for param_name, new_value in modifications.items():
        old_value = None
        
        if param_name.startswith("demand_"):
            route = param_name.replace("demand_", "").replace("_", "-")
            # Find and replace demand value
            pattern = rf'({route}):\s*(\d+)\s*passengers?/day'
            match = re.search(pattern, modified_text, re.IGNORECASE)
            if match:
                old_value = int(match.group(2))
                modified_text = re.sub(
                    rf'({route}):\s*\d+\s*passengers?/day',
                    rf'\1: {new_value} passengers/day',
                    modified_text,
                    flags=re.IGNORECASE
                )
                changes.append(f"Modified demand for {route}: {old_value} → {new_value}")
        
        elif param_name.startswith("arr_slack_") or param_name.startswith("dep_slack_"):
            airport = param_name.replace("arr_slack_", "").replace("dep_slack_", "")
            slot_type = "arrival" if "arr" in param_name else "departure"
            
            # Find the slack value in the description
            # Pattern: "ORD: max used 155, committed 152 → slack 3"
            # Need to find in the Arrivals section or Departures section
            section_keyword = "Arrivals:" if "arr" in param_name else "Departures"
            
            # Find the slack value - handle format: "      - ORD: max used 155, committed 152 → slack 3"
            # Pattern should match with optional leading dash and whitespace
            pattern = rf'(\s*[-]?\s*)({airport}):\s*max used (\d+),\s*committed (\d+)\s*→\s*slack (\d+)'
            match = re.search(pattern, modified_text, re.IGNORECASE)
            if match:
                prefix = match.group(1)  # Preserve leading whitespace and dash
                old_value = int(match.group(5))
                max_used = int(match.group(3))
                committed = int(match.group(4))
                # Adjust committed to achieve new slack
                new_committed = max_used - new_value
                # Replace using regex - preserve the prefix
                pattern_to_replace = rf'(\s*[-]?\s*)({airport}):\s*max used {max_used},\s*committed {committed}\s*→\s*slack {old_value}'
                replacement = rf'\1\2: max used {max_used}, committed {new_committed} → slack {new_value}'
                modified_text = re.sub(pattern_to_replace, replacement, modified_text, flags=re.IGNORECASE)
                changes.append(f"Modified {slot_type} slack for {airport}: {old_value} → {new_value}")
            else:
                # Try simpler pattern without prefix
                pattern = rf'({airport}):\s*max used (\d+),\s*committed (\d+)\s*→\s*slack (\d+)'
                match = re.search(pattern, modified_text, re.IGNORECASE)
                if match:
                    old_value = int(match.group(4))
                    max_used = int(match.group(2))
                    committed = int(match.group(3))
                    new_committed = max_used - new_value
                    pattern_to_replace = rf'({airport}):\s*max used {max_used},\s*committed {committed}\s*→\s*slack {old_value}'
                    replacement = rf'\1: max used {max_used}, committed {new_committed} → slack {new_value}'
                    modified_text = re.sub(pattern_to_replace, replacement, modified_text, flags=re.IGNORECASE)
                    changes.append(f"Modified {slot_type} slack for {airport}: {old_value} → {new_value}")
        
        elif param_name.startswith("availability_"):
            aircraft = param_name.replace("availability_", "").strip()
            pattern = rf'({re.escape(aircraft)}):\s*(\d+)\s*frames?\s*available'
            match = re.search(pattern, modified_text, re.IGNORECASE)
            if match:
                old_value = int(match.group(2))
                modified_text = re.sub(
                    rf'({re.escape(aircraft)}):\s*\d+\s*frames?\s*available',
                    rf'\1: {new_value} frames available',
                    modified_text,
                    flags=re.IGNORECASE
                )
                changes.append(f"Modified availability for {aircraft}: {old_value} → {new_value}")
    
    return modified_text, changes

def llm_analyze_infeasibility(problem_description, llm_response, diagnostics, client, model_name):
    """
    Use LLM to analyze infeasibility in the context of the original problem modeling.
    Returns analysis of where the infeasibility comes from in the modeling.
    """
    iis_constraints = diagnostics.get("iis_constraints", [])
    constraint_details = diagnostics.get("constraint_details", [])
    
    # Format IIS information for LLM
    iis_info = "IIS Constraints (conflicting constraints):\n"
    for constr in constraint_details:
        iis_info += f"- {constr['name']}: {constr['sense']} {constr['rhs']}\n"
        iis_info += f"  Expression: {constr['expression']}\n"
    
    analysis_prompt = f"""You are an Operations Research expert. A Gurobi optimization model was generated from a problem description, but it is INFEASIBLE.

Original Problem Description:
{problem_description}

Original Mathematical Formulation and Code (from LLM):
{llm_response}

IIS (Irreducible Infeasible Set) Analysis:
{iis_info}

Your task:
1. Analyze the IIS constraints in the context of the original problem description and mathematical formulation
2. Identify WHERE in the problem modeling the infeasibility comes from
3. Explain what constraints or parameters in the problem description are causing the conflict
4. Suggest specific modifications to the problem description that would make it feasible

Focus on:
- Which constraints in the problem description are too restrictive?
- Which parameter values need to be adjusted?
- What is the root cause of the mathematical infeasibility?

Provide a clear analysis and specific recommendations for modifying the problem description.
"""
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert Operations Research analyst specializing in diagnosing and fixing infeasible optimization models."},
                {"role": "user", "content": analysis_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error in LLM analysis: {str(e)}"

def llm_modify_problem_description(problem_description, infeasibility_analysis, client, model_name):
    """
    Use LLM to modify the problem description based on infeasibility analysis.
    Returns modified problem description and explanation of changes.
    """
    modification_prompt = f"""You are an Operations Research expert. Based on the infeasibility analysis, modify the problem description to make it feasible.

Original Problem Description:
{problem_description}

Infeasibility Analysis:
{infeasibility_analysis}

Your task:
1. Modify the problem description to address the identified infeasibility issues
2. Adjust parameter values (demands, capacities, limits, etc.) to make the problem feasible
3. Keep the problem structure and context the same - only modify numerical values
4. Make conservative but sufficient changes - adjust values enough to make it feasible
5. Preserve the natural language style and format exactly

IMPORTANT: You MUST modify at least one numerical value. Do not return the exact same description.
Return ONLY the modified problem description, maintaining the same structure and format as the original.
Do not include any explanations or commentary - just the modified problem description text.
"""
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert Operations Research analyst. Modify problem descriptions to make optimization models feasible while preserving problem structure. You must change numerical values to fix infeasibility."},
                {"role": "user", "content": modification_prompt}
            ]
        )
        modified_description = response.choices[0].message.content.strip()
        
        # Clean up the response - remove common prefixes/explanations
        # Remove markdown code blocks if present
        if modified_description.startswith("```"):
            # Extract content from code block
            lines = modified_description.split('\n')
            modified_description = '\n'.join([l for l in lines if not l.strip().startswith('```')]).strip()
        
        # Remove common explanation prefixes
        prefixes_to_remove = [
            "Here is the modified problem description:",
            "Modified Problem Description:",
            "Here's the modified version:",
            "The modified problem description:",
            "Problem Description:",
            "Modified Problem:",
        ]
        for prefix in prefixes_to_remove:
            if modified_description.startswith(prefix):
                modified_description = modified_description[len(prefix):].strip()
        
        # Try to extract just the problem description part if there are explanations
        # Look for the start of the actual description (usually starts with context)
        if "Business context" in problem_description:
            # Try to find where the actual description starts
            if "Business context" in modified_description:
                idx = modified_description.find("Business context")
                modified_description = modified_description[idx:].strip()
        
        # Remove trailing explanations
        # Common patterns that indicate explanation text
        explanation_markers = [
            "\n\nNote:",
            "\n\nThis modification",
            "\n\nThe changes",
            "\n\nI've modified",
            "\n\nChanges made:",
        ]
        for marker in explanation_markers:
            if marker in modified_description:
                modified_description = modified_description[:modified_description.find(marker)].strip()
        
        # Debug: log first 200 chars of response
        if len(modified_description) < 200:
            print(f"    DEBUG: LLM response (full): {modified_description[:500]}")
        else:
            print(f"    DEBUG: LLM response (first 200 chars): {modified_description[:200]}...")
        
        return modified_description
    except Exception as e:
        print(f"    ERROR in LLM modification: {str(e)}")
        import traceback
        traceback.print_exc()
        return problem_description  # Return original on error

def iterative_code_refinement_pipeline(problem_description, client, model_name, args, max_iterations=5):
    """
    Generic iterative refinement pipeline using LLM reasoning:
    1. Generate code from problem description
    2. Execute with auto-repair
    3. If still infeasible, use LLM to analyze IIS and identify modeling issues
    4. Use LLM to modify problem description to make it feasible
    5. Re-generate code from modified description
    6. Repeat until feasible or max iterations
    7. Save all intermediate steps
    
    This is generic and works for any optimization problem.
    """
    iteration_history = []
    current_description = problem_description
    iteration = 0
    
    # Determine output directory
    if args.problem_description:
        output_dir = os.path.dirname(os.path.abspath(args.problem_description))
    else:
        output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create iteration subdirectory
    iteration_dir = os.path.join(output_dir, "iterations")
    os.makedirs(iteration_dir, exist_ok=True)
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}")
        print(f"{'='*60}")
        
        iteration_data = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "problem_description": current_description,
            "llm_response": None,
            "extracted_code": None,
            "execution_result": None,
            "infeasibility_analysis": None,
            "problem_description_modifications": None,
            "feasible": False
        }
        
        # Generate code from current problem description
        prompt = TEMPLATE_q2mc_en.replace("{Question}", current_description.strip()).strip()
        iteration_data["prompt"] = prompt
        
        print(f"Generating code from problem description...")
        generation = generate_with_openai(
            client, prompt, model_name,
            topk=args.topk, temperature=args.temperature if args.decoding_method == "sampling" else 0,
            max_tokens=args.max_tokens
        )
        
        if not generation.outputs:
            print(f"Iteration {iteration}: Failed to generate response")
            break
        
        output = generation.outputs[0]
        iteration_data["llm_response"] = output.text
        
        # Extract code
        extracted_code = extract_code_from_output(output.text)
        iteration_data["extracted_code"] = extracted_code
        
        if not extracted_code:
            print(f"Iteration {iteration}: No code extracted")
            break
        
        # Execute with auto-repair
        print(f"Executing code with auto-repair...")
        execution_output = compile_script(
            extracted_code,
            timeout=args.timeout,
            diagnose_infeasible=args.diagnose_infeasible,
            auto_repair=args.auto_repair,
            max_repair_attempts=args.max_repair_attempts,
            problem_description=current_description
        )
        
        iteration_data["execution_result"] = execution_output
        iteration_data["repair_attempts"] = execution_output.get("repair_attempts", 0)
        iteration_data["repair_changes"] = execution_output.get("repair_changes", [])
        
        # Check if feasible
        if execution_output.get("execution_state") == "Execution Successful and Best Solution Found":
            iteration_data["feasible"] = True
            print(f"\n✓ Iteration {iteration}: SOLUTION FOUND!")
            print(f"  Objective value: {execution_output.get('execution_best_solution')}")
            iteration_history.append(iteration_data)
            break
        
        # If still infeasible, use LLM to analyze and modify problem description
        if "INFEASIBLE" in execution_output.get("execution_state", ""):
            print(f"\n✗ Iteration {iteration}: Still infeasible after code repair")
            
            # Get the current code (may have been repaired)
            current_code = execution_output.get("repaired_code") or extracted_code
            
            # Diagnose infeasibility
            print(f"  Computing IIS...")
            diagnostics = diagnose_infeasibility(current_code)
            iteration_data["infeasibility_diagnostics"] = diagnostics
            
            if not diagnostics.get("constraint_details"):
                print(f"  Could not compute IIS. Stopping refinement.")
                iteration_history.append(iteration_data)
                break
            
            # Use LLM to analyze infeasibility in modeling context
            print(f"  Using LLM to analyze infeasibility in modeling context...")
            infeasibility_analysis = llm_analyze_infeasibility(
                current_description, output.text, diagnostics, client, model_name
            )
            iteration_data["infeasibility_analysis"] = infeasibility_analysis
            print(f"  Analysis complete.")
            
            # Use LLM to modify problem description
            print(f"  Using LLM to modify problem description...")
            modified_description_raw = llm_modify_problem_description(
                current_description, infeasibility_analysis, client, model_name
            )
            iteration_data["llm_modification_response_raw"] = modified_description_raw
            
            # Check if modifications were actually made
            # First, check if it's exactly the same (before normalization)
            if modified_description_raw == current_description:
                print(f"  WARNING: LLM returned exactly identical description.")
                print(f"  This may indicate the LLM could not identify feasible modifications.")
                print(f"  Stopping refinement.")
                iteration_history.append(iteration_data)
                break
            
            # Compare normalized versions (ignore whitespace differences)
            original_normalized = re.sub(r'\s+', ' ', current_description.strip())
            modified_normalized = re.sub(r'\s+', ' ', modified_description_raw.strip())
            
            # Also check if any numbers changed (more lenient check)
            original_numbers = set(re.findall(r'\d+', current_description))
            modified_numbers = set(re.findall(r'\d+', modified_description_raw))
            numbers_changed = original_numbers != modified_numbers
            
            if modified_normalized == original_normalized and not numbers_changed:
                print(f"  WARNING: LLM returned description with no meaningful changes.")
                print(f"  Normalized text identical and no numerical values changed.")
                print(f"  Stopping refinement.")
                iteration_history.append(iteration_data)
                break
            
            modified_description = modified_description_raw
            
            # Show a brief summary of changes
            print(f"  Description modified (length: {len(current_description)} -> {len(modified_description)} chars)")
            
            # Try to identify what changed (simple diff)
            original_lines = current_description.split('\n')
            modified_lines = modified_description.split('\n')
            changes_detected = []
            for i, (orig, mod) in enumerate(zip(original_lines, modified_lines)):
                if orig.strip() != mod.strip():
                    # Extract numbers that changed
                    orig_nums = re.findall(r'\d+', orig)
                    mod_nums = re.findall(r'\d+', mod)
                    if orig_nums != mod_nums:
                        changes_detected.append(f"Line {i+1}: {orig[:60]}... -> {mod[:60]}...")
                        if len(changes_detected) >= 3:  # Limit to first 3 changes
                            break
            
            if changes_detected:
                print(f"  Detected changes:")
                for change in changes_detected[:3]:
                    print(f"    - {change}")
            else:
                print(f"  (Changes detected but not easily summarized)")
            
            iteration_data["problem_description_modifications"] = {
                "original": current_description,
                "modified": modified_description,
                "analysis": infeasibility_analysis
            }
            
            print(f"  Problem description modified.")
            current_description = modified_description
            
            # Save iteration data
            iteration_file = os.path.join(iteration_dir, f"iteration_{iteration}.json")
            with open(iteration_file, 'w', encoding='utf-8') as f:
                json.dump(iteration_data, f, indent=2, ensure_ascii=False)
            
            # Save modified problem description
            desc_file = os.path.join(iteration_dir, f"problem_description_iteration_{iteration}.txt")
            with open(desc_file, 'w', encoding='utf-8') as f:
                f.write(modified_description)
            
            iteration_history.append(iteration_data)
        else:
            # Other error, stop
            print(f"Iteration {iteration}: Execution error: {execution_output.get('execution_state')}")
            iteration_history.append(iteration_data)
            break
    
    # Save final results
    final_result = {
        "total_iterations": iteration,
        "feasible": iteration_data.get("feasible", False),
        "final_objective": execution_output.get("execution_best_solution") if iteration_data.get("feasible") else None,
        "iteration_history": iteration_history
    }
    
    # Save final modified problem description if different from original
    if current_description != problem_description:
        final_desc_file = os.path.join(output_dir, "problem_description_modified.txt")
        with open(final_desc_file, 'w', encoding='utf-8') as f:
            f.write(current_description)
        final_result["modified_problem_description_file"] = final_desc_file
        print(f"\nModified problem description saved to: {final_desc_file}")
    
    # Save iteration summary
    summary_file = os.path.join(output_dir, "refinement_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)
    print(f"Refinement summary saved to: {summary_file}")
    
    # Save final solution if feasible
    if iteration_data.get("feasible"):
        final_solution = {
            "mathematical_formulation": extract_mathematical_formulation(iteration_data["llm_response"]),
            "gurobi_code": iteration_data["extracted_code"],
            "repaired_gurobi_code": execution_output.get("repaired_code"),
            "objective_value": execution_output.get("execution_best_solution"),
            "execution_success": True,
            "execution_state": execution_output.get("execution_state"),
            "execution_result": execution_output.get("execution_result"),
            "refinement_iterations": iteration,
            "final_problem_description": current_description if current_description != problem_description else None
        }
        
        solution_file = os.path.join(output_dir, "solution.json")
        with open(solution_file, 'w', encoding='utf-8') as f:
            json.dump(final_solution, f, indent=2, ensure_ascii=False)
        print(f"Final solution saved to: {solution_file}")
    
    return final_result, iteration_data if iteration_data.get("feasible") else None

def aggressive_code_repair(code, diagnostics, pattern_type, pattern_data):
    """
    Apply more aggressive generic code repair strategies based on IIS pattern.
    This is generic and works for any optimization problem.
    """
    repaired_code = code
    changes = []
    
    constraint_details = diagnostics.get("constraint_details", [])
    
    if pattern_type == "equality_upper_bound_conflict":
        # More aggressive: Change ALL equality constraints in IIS to >=
        equality_names = pattern_data.get("equality_constraints", [])
        for eq_name in equality_names:
            # Try multiple constraint formats
            patterns = [
                rf'(model\.addConstr\([^)]*==[^)]*name\s*=\s*["\']?{re.escape(eq_name)}["\']?[^)]*\))',
                rf'(model\.addConstr\([^)]*name\s*=\s*["\']?{re.escape(eq_name)}["\']?[^)]*==[^)]*\))',
                rf'(model\.addConstr\([^)]*name\s*=\s*["\']?{re.escape(eq_name)}["\']?[^)]*sense\s*=\s*GRB\.EQUAL[^)]*\))',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, repaired_code, re.MULTILINE | re.DOTALL)
                if match:
                    old_constr = match.group(1)
                    new_constr = old_constr.replace('==', '>=').replace(' = ', ' >= ')
                    new_constr = new_constr.replace('GRB.EQUAL', 'GRB.GREATER_EQUAL')
                    repaired_code = repaired_code.replace(old_constr, new_constr)
                    changes.append(f"Changed equality constraint '{eq_name}' to >= (allow over-fulfillment)")
                    break
        
        # Also increase ALL upper bounds in IIS by larger factor
        upper_bound_names = pattern_data.get("upper_bound_constraints", [])
        for ub_name in upper_bound_names:
            patterns = [
                rf'(model\.addConstr\([^)]*<=\s*)(\d+\.?\d*)([^)]*name\s*=\s*["\']?{re.escape(ub_name)}["\']?[^)]*\))',
                rf'(model\.addConstr\([^)]*name\s*=\s*["\']?{re.escape(ub_name)}["\']?[^)]*<=\s*)(\d+\.?\d*)([^)]*\))',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, repaired_code, re.MULTILINE | re.DOTALL)
                if match:
                    old_rhs = float(match.group(2))
                    new_rhs = int(old_rhs * 2) + 2  # More aggressive: double + 2
                    old_constr = match.group(0)
                    new_constr = match.group(1) + str(new_rhs) + match.group(3)
                    repaired_code = repaired_code.replace(old_constr, new_constr)
                    changes.append(f"Increased upper bound in constraint '{ub_name}' from {old_rhs} to {new_rhs}")
                    break
    
    elif pattern_type == "multiple_upper_bounds_conflict":
        # Increase ALL conflicting bounds, not just the most restrictive
        rhs_values = pattern_data.get("rhs_values", {})
        for constr_name, old_rhs in rhs_values.items():
            patterns = [
                rf'(model\.addConstr\([^)]*<=\s*)(\d+\.?\d*)([^)]*name\s*=\s*["\']?{re.escape(constr_name)}["\']?[^)]*\))',
                rf'(model\.addConstr\([^)]*name\s*=\s*["\']?{re.escape(constr_name)}["\']?[^)]*<=\s*)(\d+\.?\d*)([^)]*\))',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, repaired_code, re.MULTILINE | re.DOTALL)
                if match:
                    new_rhs = int(old_rhs * 2) + 2
                    old_constr = match.group(0)
                    new_constr = match.group(1) + str(new_rhs) + match.group(3)
                    repaired_code = repaired_code.replace(old_constr, new_constr)
                    changes.append(f"Increased bound in constraint '{constr_name}' from {old_rhs} to {new_rhs}")
                    break
    
    elif pattern_type == "variable_bounds_conflict":
        # Remove ALL lower bounds and significantly increase upper bounds
        variable_details = diagnostics.get("variable_details", [])
        for var_detail in variable_details:
            var_name = var_detail['name']
            if var_detail.get('lower_bound_violated'):
                # Remove lower bound
                var_pattern = rf'(\w+)\s*=\s*model\.addVar\([^)]*name\s*=\s*["\']?{re.escape(var_name)}["\']?[^)]*\)'
                match = re.search(var_pattern, repaired_code)
                if match:
                    var_def = match.group(0)
                    new_def = re.sub(r',\s*lb\s*=\s*[^,)]+', '', var_def)
                    new_def = re.sub(r'lb\s*=\s*[^,)]+\s*,', '', new_def)
                    repaired_code = repaired_code.replace(var_def, new_def)
                    changes.append(f"Removed lower bound from variable '{var_name}'")
            
            if var_detail.get('upper_bound_violated'):
                # Increase or add upper bound
                var_pattern = rf'(\w+)\s*=\s*model\.addVar\([^)]*name\s*=\s*["\']?{re.escape(var_name)}["\']?[^)]*\)'
                match = re.search(var_pattern, repaired_code)
                if match:
                    var_def = match.group(0)
                    ub_match = re.search(r'ub\s*=\s*(\d+\.?\d*)', var_def)
                    if ub_match:
                        old_ub = float(ub_match.group(1))
                        new_ub = int(old_ub * 3)  # Triple the upper bound
                        new_def = var_def.replace(f"ub={old_ub}", f"ub={new_ub}")
                    else:
                        new_def = var_def.replace(')', ', ub=10000)')
                    repaired_code = repaired_code.replace(var_def, new_def)
                    changes.append(f"Increased upper bound for variable '{var_name}'")
    
    elif pattern_type == "equality_resource_conflict":
        # Change ALL equality constraints to >=
        equality_names = pattern_data.get("equality_constraints", [])
        for eq_name in equality_names:
            patterns = [
                rf'(model\.addConstr\([^)]*==[^)]*name\s*=\s*["\']?{re.escape(eq_name)}["\']?[^)]*\))',
                rf'(model\.addConstr\([^)]*name\s*=\s*["\']?{re.escape(eq_name)}["\']?[^)]*==[^)]*\))',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, repaired_code, re.MULTILINE | re.DOTALL)
                if match:
                    old_constr = match.group(1)
                    new_constr = old_constr.replace('==', '>=').replace(' = ', ' >= ')
                    repaired_code = repaired_code.replace(old_constr, new_constr)
                    changes.append(f"Changed equality constraint '{eq_name}' to >= (allow over-fulfillment)")
                    break
    
    if not changes:
        changes.append(f"Pattern '{pattern_type}' identified but no additional repairs applied")
    
    return repaired_code, changes

def compile_script(script_content, timeout=300, diagnose_infeasible=True, auto_repair=True, max_repair_attempts=3, problem_description=None):
    """
    Execute Python script and capture results using Gurobi evaluation.
    If infeasible and auto_repair is True, automatically attempts to repair the code.
    
    When auto_repair is enabled, it fixes code bugs (parameter mismatches, constraint errors)
    while preserving original constraints. Requires problem_description to compare against.
    """
    target_dir = './eval_execute'
    os.makedirs(target_dir, exist_ok=True)

    current_code = script_content
    repair_attempts = 0
    all_changes = []
    infeasibility_diagnostics = None
    
    while repair_attempts <= max_repair_attempts:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.py', dir=target_dir) as tmp_file:
            tmp_file_name = tmp_file.name
            tmp_file.write(current_code.encode())

        try:
            # Use Gurobi execution function instead of subprocess
            result = execute_gurobi(current_code)
            
            if result["success"]:
                execution_best_solution = str(result["value"])
                execution_state = "Execution Successful and Best Solution Found"
                execution_result = f"Optimal value: {result['value']}"
                
                # Add repair history if repairs were made
                if all_changes:
                    execution_result += f"\n\nAUTO-REPAIR SUCCESSFUL after {repair_attempts} attempt(s):\n"
                    for change in all_changes:
                        execution_result += f"  - {change}\n"
                
                return {
                    "execution_result": execution_result,
                    "execution_best_solution": execution_best_solution, 
                    "execution_state": execution_state,
                    "infeasibility_diagnostics": infeasibility_diagnostics,
                    "repair_attempts": repair_attempts,
                    "repair_changes": all_changes,
                    "repaired_code": current_code if repair_attempts > 0 else None
                }
            else:
                execution_best_solution = None
                execution_state = f"Execution Failed: {result['value']}"
                execution_result = result["value"]
                
                # Diagnose and repair infeasibility if model is infeasible
                if diagnose_infeasible and "INFEASIBLE" in result["value"]:
                    infeasibility_diagnostics = diagnose_infeasibility(current_code)
                    
                    if infeasibility_diagnostics and auto_repair and repair_attempts < max_repair_attempts:
                        # Attempt automatic repair
                        repaired_code, changes = repair_infeasible_code(current_code, infeasibility_diagnostics, problem_description)
                        
                        if changes and repaired_code != current_code:
                            repair_attempts += 1
                            all_changes.extend(changes)
                            current_code = repaired_code
                            
                            execution_result += f"\n\nAUTO-REPAIR ATTEMPT {repair_attempts}:\n"
                            for change in changes:
                                execution_result += f"  - {change}\n"
                            execution_result += "\nRetrying with repaired code...\n"
                            continue  # Retry with repaired code
                        else:
                            # No repair possible or repair didn't change code
                            execution_result += "\n\nINFEASIBILITY DIAGNOSTICS:\n"
                            execution_result += "=" * 60 + "\n"
                            if infeasibility_diagnostics.get("iis_constraints"):
                                execution_result += "IIS Constraints:\n"
                                for constr_name in infeasibility_diagnostics["iis_constraints"]:
                                    execution_result += f"  - {constr_name}\n"
                            if infeasibility_diagnostics.get("error"):
                                execution_result += f"\nError: {infeasibility_diagnostics['error']}\n"
                    else:
                        # Diagnostics only, no repair
                        execution_result += "\n\nINFEASIBILITY DIAGNOSTICS:\n"
                        execution_result += "=" * 60 + "\n"
                        if infeasibility_diagnostics and infeasibility_diagnostics.get("iis_constraints"):
                            execution_result += "IIS Constraints:\n"
                            for constr_name in infeasibility_diagnostics["iis_constraints"]:
                                execution_result += f"  - {constr_name}\n"
                
                # If we get here, either repair failed or max attempts reached
                if all_changes:
                    execution_result += f"\n\nAUTO-REPAIR ATTEMPTS: {repair_attempts}\n"
                    execution_result += "Changes made:\n"
                    for change in all_changes:
                        execution_result += f"  - {change}\n"
                    execution_result += "\nModel remains infeasible after repair attempts.\n"
                
                return {
                    "execution_result": execution_result,
                    "execution_best_solution": execution_best_solution, 
                    "execution_state": execution_state,
                    "infeasibility_diagnostics": infeasibility_diagnostics,
                    "repair_attempts": repair_attempts,
                    "repair_changes": all_changes,
                    "repaired_code": current_code if repair_attempts > 0 else None
                }
                    
        except Exception as e:
            execution_result = f"Execution error: {str(e)}"
            execution_best_solution = None
            execution_state = f"Execution Failed: {str(e)}"
            break
        finally:
            if os.path.exists(tmp_file_name):
                os.remove(tmp_file_name)
    
    return {
        "execution_result": execution_result,
        "execution_best_solution": execution_best_solution, 
        "execution_state": execution_state,
        "infeasibility_diagnostics": infeasibility_diagnostics,
        "repair_attempts": repair_attempts,
        "repair_changes": all_changes,
        "repaired_code": current_code if repair_attempts > 0 else None
    }

def assess_code_correctness(code, ground_truth=None, numerical_tolerance=0.05, timeout=300,
                            diagnose_infeasible=True, auto_repair=True, problem_description=None):
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
    
    # Execute the code directly using Gurobi (with automatic repair if infeasible)
    execution_output = compile_script(code, timeout=timeout, 
                                     diagnose_infeasible=diagnose_infeasible,
                                     auto_repair=auto_repair,
                                     problem_description=problem_description)
    
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
                    optimality_result = test_optimality(code, float(ground_truth))
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

def load_problem_description(description_file):
    """Load problem description from text file."""
    if not os.path.exists(description_file):
        return None
    
    with open(description_file, 'r', encoding='utf-8') as f:
        return f.read()

def generate_with_openai(client, prompt, model_name, topk=1, temperature=1, max_tokens=None):
    """Generate responses using OpenAI API."""
    try:
        # Prepare parameters, excluding max_tokens if it's None
        params = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "n": topk
        }
        
        # Handle temperature - some models don't support temperature=0
        if temperature > 0:
            params["temperature"] = temperature
        
        # Only add max_tokens if it's not None
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
            
        # Only add stop tokens if the model supports it (some models don't)
        # Note: We'll try without stop tokens first, as some models don't support this parameter
            
        response = client.chat.completions.create(**params)
        
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
    assert isinstance(args.topk, int)
    assert args.decoding_method in ["greedy", "sampling"]
    
    # Load environment variables from .env file
    load_env_file()
    
    # Initialize OpenAI client
    if args.api_key:
        client = openai.OpenAI(api_key=args.api_key)
    else:
        # Try to get API key from environment variable
        client = openai.OpenAI()
    
    print(f"Initialized OpenAI client with model: {args.model_name}")

    # Load problem description if provided
    problem_description = None
    if args.problem_description:
        problem_description = load_problem_description(args.problem_description)
        if problem_description:
            print(f"Loaded problem description from {args.problem_description}")
        else:
            print(f"Warning: Could not load problem description from {args.problem_description}")
    
    # Load test data if provided
    test_data = []
    if args.test_file:
        test_data = load_test_data(args.test_file)
        print(f"Loaded {len(test_data)} test examples")
    
    # Check if iterative refinement is enabled
    if args.iterative_refinement and problem_description:
        print("\n" + "="*60)
        print("ITERATIVE REFINEMENT MODE ENABLED")
        print("="*60)
        print("This will use LLM reasoning to:")
        print("1. Analyze infeasibility in modeling context")
        print("2. Identify root causes in problem description")
        print("3. Modify problem description to make it feasible")
        print("4. Re-generate code and retry")
        print("Continues until feasible or max iterations reached.")
        print("="*60 + "\n")
        
        # Run iterative code refinement pipeline
        final_result, final_solution = iterative_code_refinement_pipeline(
            problem_description, client, args.model_name, args, 
            max_iterations=args.max_refinement_iterations
        )
        
        if final_solution:
            print(f"\n{'='*60}")
            print("ITERATIVE REFINEMENT COMPLETED SUCCESSFULLY")
            print(f"{'='*60}")
            print(f"Total iterations: {final_result['total_iterations']}")
            print(f"Final objective: {final_result['final_objective']}")
            print(f"Solution and modified problem description saved")
            print(f"All iterations and analyses saved in iterations/ directory")
        else:
            print(f"\n{'='*60}")
            print("ITERATIVE REFINEMENT STOPPED")
            print(f"{'='*60}")
            print(f"Reached max iterations or could not make problem feasible")
            print(f"Check refinement_summary.json and iterations/ for details")
        
        return final_result
    
    # Prepare samples for evaluation
    if problem_description:
        # Use problem description file
        prompt = TEMPLATE_q2mc_en.replace("{Question}", problem_description.strip()).strip()
        sample = [{"prompt": prompt, "ground_truth": None, "source": "problem_description"}]
    elif test_data:
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
    if args.decoding_method == "greedy":
        temperature = 0
    elif args.decoding_method == "sampling":
        temperature = args.temperature
    else:
        raise ValueError("Invalid decoding method")
    
    print(f"Generation parameters - topk: {args.topk}, temperature: {temperature}")

    # Generate responses
    prompts = [example["prompt"] for example in sample]
    generations = []
    
    print("Generating responses...")
    for prompt in tqdm(prompts, desc="Generating"):
        generation = generate_with_openai(
            client, prompt, args.model_name, 
            topk=args.topk, temperature=temperature, 
            max_tokens=args.max_tokens
        )
        generations.append(generation)
    
    # Determine output directory for problem description
    problem_output_dir = None
    if args.problem_description:
        problem_output_dir = os.path.dirname(os.path.abspath(args.problem_description))
    
    # Evaluate correctness
    results = []
    overall_metrics = {
        "total_samples": 0,
        "execution_success": 0,
        "mathematical_accuracy": 0,
        "code_extraction_success": 0
    }
    
    verbose_content = []  # Collect verbose output
    problem_llm_response = None  # Store full LLM response for problem_description
    
    for example, prompt, generation in zip(sample, prompts, generations):
        outputs = generation.outputs
        
        for output in outputs:
            result_entry = {k: v for k, v in example.items()}
            result_entry["generated_output"] = output.text
            
            # Store LLM response for problem_description
            if example.get("source") == "problem_description":
                problem_llm_response = output.text
            
            # Extract mathematical formulation and code
            mathematical_formulation = extract_mathematical_formulation(output.text)
            extracted_code = extract_code_from_output(output.text)
            result_entry["mathematical_formulation"] = mathematical_formulation
            result_entry["extracted_code"] = extracted_code
            
            
            if extracted_code:
                overall_metrics["code_extraction_success"] += 1
                
                # Assess correctness
                ground_truth = example.get(args.answer_field)
                
                # Pass problem_description if available (for bug-fixing auto_repair)
                current_problem_desc = problem_description if example.get("source") == "problem_description" else None
                
                correctness_metrics, execution_output = assess_code_correctness(
                    extracted_code, ground_truth, args.numerical_tolerance, args.timeout,
                    diagnose_infeasible=args.diagnose_infeasible, auto_repair=args.auto_repair,
                    problem_description=current_problem_desc)
                
                result_entry.update(correctness_metrics)
                result_entry.update(execution_output)
                
                # Update overall metrics
                for metric in ["execution_success", "mathematical_accuracy"]:
                    if correctness_metrics[metric]:
                        overall_metrics[metric] += 1
                
                # Collect verbose output
                verbose_text = f"\n{'='*60}\n"
                verbose_text += f"Sample {overall_metrics['total_samples'] + 1}\n"
                verbose_text += f"Question: {example.get(args.question_field, 'N/A')[:100]}...\n"
                verbose_text += f"Ground Truth: {ground_truth}\n"
                verbose_text += f"Predicted Solution: {execution_output.get('execution_best_solution', 'N/A')}\n"
                verbose_text += f"Correctness Metrics: {correctness_metrics}\n"
                verbose_text += f"Execution State: {execution_output.get('execution_state', 'N/A')}\n"
                verbose_text += f"{'='*60}\n"
                verbose_content.append(verbose_text)
                
                # Print verbose output
                if args.verbose:
                    print(verbose_text)
                
                # Save problem-specific outputs if using problem_description
                if problem_output_dir and example.get("source") == "problem_description":
                    # Save JSON with formulation, code, and objective
                    problem_result = {
                        "mathematical_formulation": mathematical_formulation,
                        "gurobi_code": extracted_code,
                        "objective_value": execution_output.get("execution_best_solution"),
                        "execution_success": correctness_metrics.get("execution_success", False),
                        "execution_state": execution_output.get("execution_state", "Unknown"),
                        "execution_result": execution_output.get("execution_result", "")
                    }
                    
                    # Add infeasibility diagnostics and repair info if available
                    if execution_output.get("infeasibility_diagnostics"):
                        problem_result["infeasibility_diagnostics"] = execution_output["infeasibility_diagnostics"]
                    if execution_output.get("repair_attempts", 0) > 0:
                        problem_result["repair_attempts"] = execution_output["repair_attempts"]
                        problem_result["repair_changes"] = execution_output.get("repair_changes", [])
                        if execution_output.get("repaired_code"):
                            problem_result["repaired_gurobi_code"] = execution_output["repaired_code"]
                    
                    json_path = os.path.join(problem_output_dir, "solution.json")
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(problem_result, f, indent=2, ensure_ascii=False)
                    print(f"Solution saved to {json_path}")
            else:
                # No code found
                result_entry.update({
                    "execution_success": False,
                    "mathematical_accuracy": False,
                    "execution_result": "No code found",
                    "execution_best_solution": None,
                    "execution_state": "No code found"
                })
                
                verbose_text = f"\n{'='*60}\n"
                verbose_text += f"Sample {overall_metrics['total_samples'] + 1}\n"
                verbose_text += f"Error: No code found in output\n"
                verbose_text += f"{'='*60}\n"
                verbose_content.append(verbose_text)
                
                if args.verbose:
                    print(verbose_text)
                
                # Save problem-specific outputs even if no code found
                if problem_output_dir and example.get("source") == "problem_description":
                    problem_result = {
                        "mathematical_formulation": mathematical_formulation,
                        "gurobi_code": None,
                        "objective_value": None,
                        "execution_success": False,
                        "execution_state": "No code found",
                        "execution_result": "No code found in LLM output"
                    }
                    
                    json_path = os.path.join(problem_output_dir, "solution.json")
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(problem_result, f, indent=2, ensure_ascii=False)
                    print(f"Solution (with errors) saved to {json_path}")
            
            overall_metrics["total_samples"] += 1
            results.append(result_entry)
    
    # Save verbose output for problem_description after all processing
    if problem_output_dir and problem_llm_response:
        verbose_path = os.path.join(problem_output_dir, "verbose_output.txt")
        with open(verbose_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("VERBOSE EVALUATION OUTPUT\n")
            f.write("="*60 + "\n\n")
            f.write("Full LLM Response:\n")
            f.write("-"*60 + "\n")
            f.write(problem_llm_response)
            f.write("\n\n")
            f.write("="*60 + "\n")
            f.write("EVALUATION DETAILS\n")
            f.write("="*60 + "\n")
            for v in verbose_content:
                f.write(v)
        print(f"Verbose output saved to {verbose_path}")
    
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
    parser = argparse.ArgumentParser(description="Evaluate optimization code generation with OpenAI models")
    parser.add_argument("--model_name", type=str, required=True, 
                       help="OpenAI model name (e.g., 'gpt-4', 'gpt-3.5-turbo', 'gpt-5')")
    parser.add_argument("--api_key", type=str, default=None, 
                       help="OpenAI API key (if not provided, will use OPENAI_API_KEY env var)")
    parser.add_argument("--topk", type=int, default=1, help="Number of responses to generate per prompt")
    parser.add_argument("--decoding_method", type=str, default="sampling", choices=["greedy", "sampling"], 
                       help="Decoding method")
    parser.add_argument("--temperature", type=float, default=1, help="Temperature for sampling")
    parser.add_argument("--max_tokens", type=int, default=None, help="Maximum tokens to generate")
    
    # Input/Output files
    parser.add_argument("--problem_description", type=str, default=None, 
                       help="Path to problem description text file (e.g., problem_description.txt)")
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
    parser.add_argument("--timeout", type=int, default=300, 
                       help="Timeout for code execution in seconds")
    parser.add_argument("--numerical_tolerance", type=float, default=0.05, 
                       help="Tolerance for numerical accuracy comparison")
    parser.add_argument("--verbose", action="store_true", 
                       help="Print detailed evaluation information")
    parser.add_argument("--diagnose_infeasible", action="store_true", default=True,
                       help="Automatically diagnose infeasible models using IIS")
    parser.add_argument("--auto_repair", action="store_true", default=True,
                       help="Automatically repair infeasible models based on IIS analysis")
    parser.add_argument("--max_repair_attempts", type=int, default=3,
                       help="Maximum number of automatic repair attempts")
    parser.add_argument("--iterative_refinement", action="store_true",
                       help="Enable iterative code refinement: aggressively repair code if still infeasible after initial repair")
    parser.add_argument("--max_refinement_iterations", type=int, default=5,
                       help="Maximum number of iterative refinement iterations")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
