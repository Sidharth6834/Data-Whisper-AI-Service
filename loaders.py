import os
import json
import pandas as pd
from typing import Union, IO, List, Tuple, Optional

# Security and performance boundaries
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_ROW_LIMIT = 200000                 # 200,000 rows

def get_excel_sheets(file_obj: Union[str, IO]) -> List[str]:
    """
    Retrieves the list of sheet names from an Excel file.
    """
    try:
        excel_file = pd.ExcelFile(file_obj)
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
        return excel_file.sheet_names
    except Exception as e:
        raise ValueError(f"Failed to read Excel sheet names: {str(e)}")

def load_file(
    file_obj: Union[str, IO], 
    file_name: str = "", 
    sheet_name: Optional[str] = None
) -> Tuple[pd.DataFrame, bool]:
    """
    Loads an uploaded file (CSV, Excel, or JSON) into a pandas DataFrame.
    Enforces file size and row limitations.
    """
    # 1. Enforce file size limit at input boundary
    file_size = 0
    if isinstance(file_obj, str):
        if os.path.exists(file_obj):
            file_size = os.path.getsize(file_obj)
    else:
        file_size = getattr(file_obj, "size", 0)
        
    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File size ({file_size / (1024*1024):.1f}MB) exceeds the safety limit of {MAX_FILE_SIZE_BYTES / (1024*1024):.0f}MB.")

    name = file_name if file_name else (file_obj if isinstance(file_obj, str) else "")
    if not name:
        raise ValueError("File name or file path must be provided to determine the format.")
        
    ext = os.path.splitext(name)[1].lower()
    df = None
    was_flattened = False
    
    try:
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)

        if ext == '.csv':
            df = pd.read_csv(file_obj)
            
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_obj, sheet_name=sheet_name)
            
        elif ext == '.json':
            if isinstance(file_obj, str):
                with open(file_obj, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                content = file_obj.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                if hasattr(file_obj, 'seek'):
                    file_obj.seek(0)
                    
            if not content.strip():
                raise ValueError("JSON file is empty.")
                
            raw_data = json.loads(content)
            
            if isinstance(raw_data, dict):
                data_list = [raw_data]
            elif isinstance(raw_data, list):
                data_list = raw_data
            else:
                raise ValueError("Invalid JSON structure. Expected a dictionary or a list of dictionaries.")
                
            is_nested = False
            for item in data_list:
                if isinstance(item, dict):
                    for val in item.values():
                        if isinstance(val, (dict, list)):
                            is_nested = True
                            break
                if is_nested:
                    break
                    
            if is_nested:
                df = pd.json_normalize(data_list)
                was_flattened = True
            else:
                df = pd.DataFrame(data_list)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Please upload a CSV, Excel, or JSON file.")
            
    except Exception as e:
        from pandas.errors import EmptyDataError
        if isinstance(e, EmptyDataError) or "No columns to parse" in str(e):
            raise ValueError(f"The file is empty: {str(e)}")
        if isinstance(e, ValueError):
            raise e
        raise ValueError(f"Failed to parse file: {str(e)}")
        
    # 2. Enforce row counts limit
    if df is None or df.empty or df.shape[0] == 0:
        raise ValueError("The uploaded file has no rows or is empty.")
        
    if df.shape[0] > MAX_ROW_LIMIT:
        raise ValueError(f"Dataset has {df.shape[0]:,} rows, which exceeds the safety limit of {MAX_ROW_LIMIT:,} rows.")
        
    return df, was_flattened
