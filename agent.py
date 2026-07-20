import os
import pandas as pd
from typing import Dict, Any, Optional

# Import LangChain agent toolkit and models
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# Import LangChain code execution tools to apply security patches
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_experimental.utilities.python import PythonREPL
from sandbox import PythonSandbox

# Configure headless matplotlib backend for web safety via environment variable
os.environ['MPLBACKEND'] = 'Agg'

def clean_code_query(query: str) -> str:
    query = query.strip()
    
    # Strip markdown code block wrappers (e.g. ```python ... ```)
    if query.startswith("```"):
        newline_idx = query.find("\n")
        if newline_idx != -1:
            query = query[newline_idx+1:]
        else:
            query = query[3:]
        if query.endswith("```"):
            query = query[:-3]
    
    # Strip single backticks or outer quotes
    if query.startswith("`") and query.endswith("`"):
        query = query[1:-1]
    elif query.startswith("'") and query.endswith("'"):
        query = query[1:-1]
    elif query.startswith('"') and query.endswith('"'):
        query = query[1:-1]
        
    return query.strip()

# Monkeypatch LangChain's execution tools to run within our secure PythonSandbox
def secure_ast_repl_run(self, query: str, run_manager: Optional[Any] = None) -> str:
    query = clean_code_query(query)
    locals_dict = getattr(self, "locals", None)
    sandbox = PythonSandbox(locals_dict=locals_dict)
    res = sandbox.run(query, timeout=5.0)
    
    if not res["success"]:
        # Halt execution immediately on security violations or timeouts
        if "Security Violation" in res["stderr"] or "Timeout Error" in res["stderr"]:
            raise ValueError(res["stderr"])
        # Return standard coding errors to let the agent self-correct
        return res["stderr"]
    return res["stdout"]

def secure_python_repl_run(self, command: str) -> str:
    command = clean_code_query(command)
    locals_dict = getattr(self, "locals", None)
    sandbox = PythonSandbox(locals_dict=locals_dict)
    res = sandbox.run(command, timeout=5.0)
    
    if not res["success"]:
        if "Security Violation" in res["stderr"] or "Timeout Error" in res["stderr"]:
            raise ValueError(res["stderr"])
        return res["stderr"]
    return res["stdout"]

# Apply the patches globally
PythonAstREPLTool._run = secure_ast_repl_run
PythonREPL.run = secure_python_repl_run

def detect_chart_intent(question: str) -> bool:
    """
    Detects if the user's question implies a chart should be generated.
    """
    keywords = [
        "plot", "chart", "graph", "bar", "line", "pie", 
        "compare", "trend", "distribution", "breakdown",
        "histogram", "scatter", "visualise", "visualize"
    ]
    question_lower = question.lower()
    return any(kw in question_lower for kw in keywords)

def create_agent(
    dataframe: pd.DataFrame, 
    api_key: str, 
    provider: str = "OpenAI", 
    model_name: str = "gpt-4o"
) -> Any:
    """
    Initialises and returns a LangChain pandas dataframe agent.
    """
    if not api_key:
        if provider == "Gemini":
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        elif provider == "Groq":
            api_key = os.getenv("GROQ_API_KEY")
        else:
            env_var = f"{provider.upper()}_API_KEY"
            api_key = os.getenv(env_var, "")
        
    if not api_key:
        raise ValueError(f"API key for {provider} is required but was not provided or found in the environment.")

    if provider == "OpenAI":
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.0
        )
        agent_type = "openai-tools"
    elif provider == "Anthropic":
        llm = ChatAnthropic(
            model=model_name,
            api_key=api_key,
            temperature=0.0
        )
        agent_type = "tool-calling"
    elif provider == "Gemini":
        # Map deprecated 1.5 models to active 2.0 / 2.5 versions to avoid API 404s
        resolved_model = model_name
        if "1.5" in model_name:
            if "flash" in model_name:
                resolved_model = "gemini-2.0-flash"
            else:
                resolved_model = "gemini-2.5-pro"
        llm = ChatGoogleGenerativeAI(
            model=resolved_model,
            google_api_key=api_key,
            temperature=0.0
        )
        agent_type = "tool-calling"
    elif provider == "Groq":
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.0
        )
        agent_type = "zero-shot-react-description"
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Supported: OpenAI, Anthropic, Gemini, Groq")

    prefix_instructions = (
        "You are working with a pandas dataframe in Python. The name of the dataframe is `df`.\n\n"
        "CRITICAL SYSTEM REQUIREMENT: Do NOT under any circumstances redefine, overwrite, or mock the `df` variable "
        "(e.g., do not write code like `df = pd.DataFrame(...)` using hardcoded list/dict data). "
        "The `df` variable is already fully loaded in your execution environment with the complete dataset. "
        "Always query, filter, and process the pre-loaded `df` directly, and perform all actions "
        "(such as prints or matplotlib plotting) using this pre-loaded variable.\n\n"
        "CRITICAL: You MUST base your final answer and summary numbers directly and strictly on the output/stdout of the Python execution tool. "
        "For example, if the tool outputs a count of 6322, your final response must report 6322. Never make up, guess, or hallucinate "
        "different numbers than what the Python execution tool printed. If the tool output is a number, use that exact number in your response.\n\n"
        "CRITICAL FOR FILTERING & EXPORTS: If the user's question involves filtering, slicing, or finding specific rows in the dataset "
        "(e.g., 'how many rows have total_amount > 5000', 'list all electronics sales', etc.), you must make sure that the final expression "
        "evaluated in your Python code is the filtered DataFrame itself (e.g. `df[df['Total_Amount'] > 5000]`). "
        "Do not just print the length or print a count. Evaluate the filtered DataFrame itself so the system can capture the rows and "
        "allow the user to download the filtered CSV. You can print the count as well, but the final expression must be the filtered DataFrame.\n\n"
        "IMPORTANT: Column names and filter values are strictly case-sensitive. Always verify the exact capitalization of columns "
        "(e.g., use 'Product' instead of 'product', 'Payment_Method' instead of 'payment_method') before writing code to avoid KeyErrors.\n\n"
        "NOTE: If the Python execution tool outputs a single column name (like 'product' or 'Product'), it indicates a KeyError occurred. "
        "Re-examine the exact column names of `df` and correct your code."
    )

    agent = create_pandas_dataframe_agent(
        llm,
        dataframe,
        verbose=True,
        agent_type=agent_type,
        return_intermediate_steps=True,
        allow_dangerous_code=True,
        prefix=prefix_instructions
    )
    return agent

def is_dummy_df_definition(node) -> bool:
    import ast
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == 'df':
                val = node.value
                if isinstance(val, ast.Call):
                    func_name = ""
                    if isinstance(val.func, ast.Name):
                        func_name = val.func.id
                    elif isinstance(val.func, ast.Attribute) and isinstance(val.func.value, ast.Name):
                        if val.func.value.id == 'pd':
                            func_name = val.func.attr
                    if func_name == 'DataFrame':
                        if val.args and isinstance(val.args[0], (ast.Dict, ast.List)):
                            return True
                        if val.keywords:
                            return True
    return False

def extract_dataframe_from_code(code_str: str, df_input: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Statically and dynamically analyzes the executed python code to extract
    the full resulting DataFrame.
    """
    if not code_str:
        return None
        
    code_str = clean_code_query(code_str)
        
    import ast
    import numpy as np
    
    try:
        tree = ast.parse(code_str)
        if not tree.body:
            return None
            
        # Filter out dummy/mock DataFrame redefinitions (e.g. df = pd.DataFrame(...))
        tree.body = [node for node in tree.body if not is_dummy_df_definition(node)]
        if not tree.body:
            return None
            
        # Recursive helper to strip print(), len(), .head(), .tail(), .to_markdown() etc.
        def get_base_node(node: ast.AST) -> ast.AST:
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id == 'print' and node.args:
                        return get_base_node(node.args[0])
                    elif node.func.id == 'len' and node.args:
                        return get_base_node(node.args[0])
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ('head', 'tail', 'to_markdown', 'to_string', 'to_csv', 'to_excel', 'to_html', 'describe'):
                        return get_base_node(node.func.value)
            return node

        # Intercept prints to capture DataFrames
        captured_from_print = []
        original_print = print
        def custom_print(*args, **kwargs):
            for arg in args:
                if isinstance(arg, (pd.DataFrame, pd.Series)):
                    captured_from_print.append(arg)
            original_print(*args, **kwargs)

        # Setup local environment for execution
        local_vars = {
            "df": df_input.copy(),
            "pd": pd,
            "np": np,
            "print": custom_print
        }
        
        # Process the last statement
        last_node = tree.body[-1]
        captured_df = None
        
        if isinstance(last_node, ast.Expr):
            # Strip preview/formatting calls from the expression
            base_val = get_base_node(last_node.value)
            # Assign the base expression to ___result___
            temp_assign = ast.Assign(
                targets=[ast.Name(id='___result___', ctx=ast.Store())],
                value=base_val
            )
            ast.copy_location(temp_assign, last_node)
            tree.body[-1] = temp_assign
            
            # Execute the AST
            ast.fix_missing_locations(tree)
            exec(compile(tree, filename="<string>", mode="exec"), {}, local_vars)
            res = local_vars.get("___result___")
            if isinstance(res, (pd.DataFrame, pd.Series)):
                captured_df = res.to_frame() if isinstance(res, pd.Series) else res
        else:
            # Just execute the AST
            ast.fix_missing_locations(tree)
            exec(compile(tree, filename="<string>", mode="exec"), {}, local_vars)
            
        # Fallbacks:
        # 1. If we captured a DataFrame from custom_print
        if captured_df is None and captured_from_print:
            res = captured_from_print[-1]
            captured_df = res.to_frame() if isinstance(res, pd.Series) else res
            
        # 2. If a new DataFrame variable was created in locals
        if captured_df is None:
            for var_name, var_val in list(local_vars.items()):
                if var_name not in ("df", "___result___") and isinstance(var_val, pd.DataFrame):
                    captured_df = var_val
                    break
                    
        return captured_df
    except Exception as e:
        return None

def ask_question(agent: Any, question: str, dataframe: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Runs a query against the pandas agent and extracts response text, generated code, and Matplotlib figures.
    """
    # Reset df in tool locals to ensure any past queries that overwrote or re-assigned df do not persist
    if dataframe is not None:
        for tool in getattr(agent, "tools", []):
            if hasattr(tool, "locals") and isinstance(tool.locals, dict) and "df" in tool.locals:
                tool.locals["df"] = dataframe
            if hasattr(tool, "globals") and isinstance(tool.globals, dict) and "df" in tool.globals:
                tool.globals["df"] = dataframe

    import matplotlib.pyplot as plt
    plt.close('all')
    
    has_chart_intent = detect_chart_intent(question)
    
    modified_question = question
    if has_chart_intent:
        modified_question += (
            "\n\nIMPORTANT: The user question implies a visual chart/plot. "
            "Please generate an appropriate Matplotlib chart (bar, line, pie, or histogram) "
            "using pandas plotting or matplotlib.pyplot, and make sure to display it. "
            "Do not call plt.show() if it is blocked, but ensure the plot is drawn to the current figure."
        )

    try:
        response = agent.invoke({"input": modified_question})
        answer = response.get("output", "")
        
        # Extract code from intermediate steps
        intermediate_steps = response.get("intermediate_steps", [])
        code_snippets = []
        
        for action, observation in intermediate_steps:
            if hasattr(action, "tool") and action.tool in ["python_repl_ast", "python_repl"]:
                tool_input = action.tool_input
                if isinstance(tool_input, dict):
                    code_snippet = tool_input.get("query", tool_input.get("code", str(tool_input)))
                else:
                    code_snippet = str(tool_input)
                code_snippets.append(code_snippet.strip())
                
        generated_code = "\n\n".join(code_snippets) if code_snippets else ""
        
        # Capture any generated Matplotlib figures safely
        chart_fig = None
        try:
            if plt.get_fignums():
                temp_fig = plt.gcf()
                # Only capture the figure if it actually contains axes with plotted data
                if temp_fig.axes:
                    chart_fig = temp_fig
                plt.close('all')
        except Exception as plot_err:
            print(f"Chart extraction warning: {str(plot_err)}")
        
        # Extract the DataFrame resulting from the query if possible
        captured_df = None
        if dataframe is not None and code_snippets:
            for snippet in reversed(code_snippets):
                try:
                    df_extracted = extract_dataframe_from_code(snippet, dataframe)
                    if df_extracted is not None:
                        captured_df = df_extracted
                        break
                except Exception:
                    continue
            
        return {
            "answer": answer,
            "generated_code": generated_code,
            "chart": chart_fig,
            "captured_df": captured_df,
            "error": None
        }
    except Exception as e:
        return {
            "answer": "An error occurred while processing your query.",
            "generated_code": "",
            "chart": None,
            "captured_df": None,
            "error": str(e)
        }
    finally:
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception:
            pass
