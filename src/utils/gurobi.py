# gurobipy
from gurobipy import GRB
import gurobipy as gp
import os
from contextlib import redirect_stdout, redirect_stderr


def execute_gurobi(code_str, timeout=None):
    """Execute Gurobi code and return optimal value or error type.
    
    Args:
        code_str: Python code string to execute
        timeout: Timeout in seconds (sets Gurobi TimeLimit parameter)
    """
    try:
        exec_globals = {'gp': gp, 'GRB': GRB}
        with open(os.devnull, 'w') as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            exec(code_str, exec_globals)
            
            # Try to call common function names that might contain the optimization
            function_names = ['main', 'solve_problem', 'solve', 'optimize', 'run']
            for func_name in function_names:
                if func_name in exec_globals and callable(exec_globals[func_name]):
                    try:
                        # Check if function needs data parameter
                        import inspect
                        sig = inspect.signature(exec_globals[func_name])
                        if len(sig.parameters) > 0:
                            # Function needs parameters, try to find data in globals
                            if 'data' in exec_globals:
                                exec_globals[func_name](exec_globals['data'])
                            else:
                                # Try to create default data from the code
                                continue
                        else:
                            exec_globals[func_name]()
                        break
                    except Exception as e:
                        # If function call fails, continue trying other functions
                        continue
        
        # Get the model from exec_globals - prioritize 'model' variable as specified in prompt
        model = None
        
        # First try 'model' (as specified in the prompt)
        if 'model' in exec_globals:
            potential_model = exec_globals['model']
            if hasattr(potential_model, 'optimize') and hasattr(potential_model, 'status'):
                model = potential_model
        
        # If not found, try other common variable names
        if model is None:
            for var_name in ['m', 'Model', 'M']:
                if var_name in exec_globals:
                    potential_model = exec_globals[var_name]
                    if hasattr(potential_model, 'optimize') and hasattr(potential_model, 'status'):
                        model = potential_model
                        break
        
        # If still no model found, try to find any Gurobi model in the namespace
        if model is None:
            for var_name, var_value in exec_globals.items():
                if hasattr(var_value, 'optimize') and hasattr(var_value, 'status') and hasattr(var_value, 'objVal'):
                    model = var_value
                    break
        
        if model is None:
            return {"success": False, "value": "No Gurobi model found in executed code"}
        
        # Set time limit and other parameters BEFORE optimization (if not already optimized)
        # Also set other parameters to prevent hanging
        if timeout is not None:
            try:
                model.setParam('TimeLimit', timeout)
                model.setParam('OutputFlag', 0)  # Suppress output
                # Set a reasonable MIP gap to prevent long optimization
                try:
                    model.setParam('MIPGap', 0.01)  # 1% gap tolerance
                except:
                    pass
            except:
                pass  # Parameters might already be set
        
        # Check if model solved successfully
        if model.status == GRB.OPTIMAL:
            return {"success": True, "value": model.objVal}
        else:
            # Model didn't solve optimally
            status_map = {
                GRB.INFEASIBLE: "INFEASIBLE",
                GRB.UNBOUNDED: "UNBOUNDED", 
                GRB.INF_OR_UNBD: "INF_OR_UNBD",
                GRB.TIME_LIMIT: "TIME_LIMIT"
            }
            status_str = status_map.get(model.status, f"STATUS_{model.status}")
            return {"success": False, "value": f"Model not optimal: {status_str}"}
            
    except Exception as e:
        return {"success": False, "value": str(e)}
    

def test_equivalence(computed_value, ground_truth, tolerance=1e-3):
    """Test if computed solution matches ground truth using relative error.
    
    Args:
        computed_value: Computed solution value
        ground_truth: Ground truth value (used as denominator for relative error)
        tolerance: Relative error tolerance (default: 1e-3 = 0.1%)
    
    Returns:
        True if relative error < tolerance, False otherwise
    """
    # Compute relative error: |computed - ground_truth| / |ground_truth|
    if abs(ground_truth) == 0:
        # Ground truth is zero, check if computed is also zero
        return abs(computed_value) < tolerance
    relative_error = abs(computed_value - ground_truth) / abs(ground_truth)
    return relative_error < tolerance
    
    
def test_optimality(code, ground_truth, timeout=None, tolerance=1e-3):
    """Test if the code produces the correct optimal solution.
    
    Args:
        code: Python code string to execute
        ground_truth: Expected optimal value (float)
        timeout: Timeout in seconds for code execution
        tolerance: Relative error tolerance (default: 1e-3 = 0.1%)
    
    Returns:
        "correct" if the solution matches ground truth within tolerance
        "wrong_answer" if the solution doesn't match
        "runtime_error" if code execution fails
    """
    result = execute_gurobi(code, timeout=timeout)
    if not result["success"]:
        return "runtime_error"
    if test_equivalence(result["value"], ground_truth, tolerance=tolerance):
        return "correct"
    else:
        return "wrong_answer"
