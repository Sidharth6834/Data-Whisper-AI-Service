"""
DataWhisperer Python Code Sandbox.

ATTACK VECTORS DEFENDED:
1. Filesystem Access: Blocks built-in 'open()' and imports of standard libraries like 'os', 'sys', 'subprocess', 'shutil'.
2. Network Operations: Blocks network-bound libraries like 'requests', 'urllib', 'socket', 'http'.
3. Dynamic Code Execution: Blocks built-ins 'eval()', 'exec()', and 'compile()'.
4. Reflection/Sandbox Escapes: Blocks accessing double-underscore variables (dunders) such as '__globals__', 
   '__subclasses__', '__code__', and '__class__' which are commonly used to traverse python object hierarchies 
   and retrieve forbidden builtins.
5. Infinite Loops & CPU Hangs: Implements a hard thread-based timeout (default 5 seconds) to release the main execution thread.

LIMITATIONS (WHAT IT DOES NOT DEFEND AGAINST):
1. CPU Resource Consumption: Thread-based timeouts on Windows cannot force-terminate CPU-bound worker threads 
   (e.g., 'while True: pass') due to Python's GIL. The worker thread will continue running in the background until the 
   Streamlit app process is restarted.
2. Memory Exhaustion: Statically allowed modules like 'numpy' or 'pandas' could still allocate massive objects (e.g. 
   'np.zeros((1000000, 1000000))') leading to out-of-memory crashes on the host system.
3. Logical Data Modification: While code is sandboxed, the DataFrame loaded in memory is passed by reference, so the 
   executed code can drop columns or modify values in the active session state if not reloaded or protected.
"""

import sys
import io
import ast
import threading
from typing import Dict, Any, Optional, List

# Define the allowlist for imports
ALLOWED_MODULES = {'pandas', 'numpy', 'matplotlib', 'math', 'datetime', 'tabulate'}

class SafetyVisitor(ast.NodeVisitor):
    """
    AST Visitor that statically analyzes python code and records security violations.
    """
    def __init__(self, allowlist: set):
        self.allowlist = allowlist
        self.errors: List[str] = []

    def visit_Import(self, node):
        for name in node.names:
            base_module = name.name.split('.')[0]
            if base_module not in self.allowlist:
                self.errors.append(f"Import of module '{name.name}' is forbidden.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            base_module = node.module.split('.')[0]
            if base_module not in self.allowlist:
                self.errors.append(f"Import from module '{node.module}' is forbidden.")
        self.generic_visit(node)

    def visit_Call(self, node):
        # Block forbidden built-in functions
        forbidden_funcs = {'eval', 'exec', 'compile', 'open', 'getattr', 'setattr', 'globals', 'locals', '__import__'}
        if isinstance(node.func, ast.Name):
            if node.func.id in forbidden_funcs:
                self.errors.append(f"Function call '{node.func.id}' is forbidden for security reasons.")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # Block reflection techniques and double-underscore attributes (sandbox escapes)
        forbidden_attrs = {'__globals__', '__subclasses__', '__code__', '__builtins__', '__import__', '__class__', '__mro__'}
        if node.attr in forbidden_attrs:
            self.errors.append(f"Access to attribute '{node.attr}' is forbidden.")
        elif node.attr.startswith('__') and node.attr.endswith('__'):
            self.errors.append(f"Access to double-underscore attribute '{node.attr}' is forbidden.")
        self.generic_visit(node)

def validate_code(code_string: str) -> Optional[str]:
    """
    Statically analyzes python code using Python's AST module to block unsafe operations.
    
    Args:
        code_string: The code text to validate.
        
    Returns:
        Optional[str]: Safety error description if violations exist, else None.
    """
    try:
        tree = ast.parse(code_string)
    except SyntaxError as e:
        return f"Syntax Error: {str(e)}"
        
    visitor = SafetyVisitor(ALLOWED_MODULES)
    visitor.visit(tree)
    
    if visitor.errors:
        return "Security Violation: " + " | ".join(visitor.errors)
    return None

class PythonSandbox:
    """
    A safety sandbox executing Python code inside isolated thread contexts,
    performing AST validation and enforcing execution timeouts.
    """
    def __init__(self, locals_dict: Optional[Dict[str, Any]] = None):
        self.locals = locals_dict if locals_dict is not None else {}
        # Prepopulate namespace with required libraries
        if 'pd' not in self.locals:
            import pandas as pd
            self.locals['pd'] = pd
        if 'plt' not in self.locals:
            import matplotlib.pyplot as plt
            self.locals['plt'] = plt

    def run(self, code: str, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Runs Python code inside a timed thread with AST safety checks.
        Allows returning the evaluation of the last expression (Jupyter behavior).
        """
        # 1. Perform static analysis validation
        validation_error = validate_code(code)
        if validation_error:
            return {
                "stdout": "",
                "stderr": validation_error,
                "success": False
            }

        # 2. Set up local redirect buffers for output capturing
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        
        # Thread-safe print override in the execution scope
        def custom_print(*args, **kwargs):
            file = kwargs.pop('file', stdout_buf)
            if file == sys.stdout:
                file = stdout_buf
            elif file == sys.stderr:
                file = stderr_buf
                
            import builtins
            builtins.print(*args, file=file, **kwargs)

        self.locals['print'] = custom_print

        error_container = []
        result_container = []

        def worker_target():
            try:
                # Parse AST
                tree = ast.parse(code)
                if not tree.body:
                    return
                
                last_node = tree.body[-1]
                if isinstance(last_node, ast.Expr):
                    # Execute all preceding statements
                    if len(tree.body) > 1:
                        exec_nodes = ast.Module(body=tree.body[:-1], type_ignores=[])
                        exec(compile(exec_nodes, filename="<sandbox>", mode="exec"), self.locals, self.locals)
                    
                    # Evaluate the final expression statement
                    eval_node = ast.Expression(body=last_node.value)
                    val = eval(compile(eval_node, filename="<sandbox>", mode="eval"), self.locals, self.locals)
                    if val is not None:
                        result_container.append(val)
                else:
                    # Execute entire block
                    exec(compile(tree, filename="<sandbox>", mode="exec"), self.locals, self.locals)
            except Exception as e:
                error_container.append(e)
                stderr_buf.write(str(e))

        worker_thread = threading.Thread(target=worker_target)
        worker_thread.daemon = True
        worker_thread.start()
        worker_thread.join(timeout=timeout)

        # Check for timeout
        if worker_thread.is_alive():
            timeout_msg = f"Timeout Error: Code execution exceeded the limit of {timeout} seconds."
            return {
                "stdout": stdout_buf.getvalue(),
                "stderr": timeout_msg,
                "success": False
            }

        stdout_val = stdout_buf.getvalue()
        stderr_val = stderr_buf.getvalue()
        
        # If code ran successfully, had no prints, but returned an expression value, capture it
        if not error_container and not stdout_val and result_container:
            stdout_val = str(result_container[0])

        return {
            "stdout": stdout_val,
            "stderr": stderr_val,
            "success": not error_container
        }
