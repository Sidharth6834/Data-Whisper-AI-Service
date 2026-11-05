from dotenv import load_dotenv
import os
# Load environment variables from the root .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import shutil
import io
import base64
from typing import Optional
import uuid

# Local imports
from loaders import load_file, get_excel_sheets
from agent import create_agent, ask_question

app = FastAPI(title="DataWhisperer AI Service")

# Enable CORS for communication from Node.js backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/")
def read_root():
    return {"status": "healthy", "service": "DataWhisperer AI Service"}

# Store active dataframes in memory keyed by session_id
dataframes = {}

# Temp directory to write uploaded files for processing
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/load")
async def load_dataset(
    file: UploadFile = File(...),
    sheet_name: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None)
):
    if not session_id or session_id == "null":
        session_id = str(uuid.uuid4())
        
    file_ext = os.path.splitext(file.filename)[1].lower()
    temp_path = os.path.join(TEMP_DIR, f"{session_id}{file_ext}")
    
    try:
        # Write to temporary file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # If it's Excel, check sheet list
        sheets = []
        if file_ext in [".xlsx", ".xls"]:
            sheets = get_excel_sheets(temp_path)
            # If multiple sheets exist and no sheet_name was chosen, return the sheet list first
            if len(sheets) > 1 and not sheet_name:
                return {
                    "session_id": session_id,
                    "requires_sheet_selection": True,
                    "sheets": sheets
                }
                
        # Load the file
        df, was_flattened = load_file(temp_path, file.filename, sheet_name=sheet_name)
        
        # Save DataFrame to in-memory dictionary
        dataframes[session_id] = df
        
        # Clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        # Return summary/metadata
        return {
            "session_id": session_id,
            "requires_sheet_selection": False,
            "filename": file.filename,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "columns_list": list(df.columns),
            "was_flattened": was_flattened,
            "preview": df.head(5).to_dict(orient="records")
        }
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/query")
async def query_dataset(
    session_id: str = Form(...),
    question: str = Form(...),
    provider: str = Form(...),
    model_name: str = Form(...),
    api_key: Optional[str] = Form("")
):
    if session_id not in dataframes:
        raise HTTPException(status_code=404, detail="Dataset not found or session expired.")
        
    df = dataframes[session_id]
    
    try:
        # Instantiate/Retrieve LangChain Agent
        agent = create_agent(df, provider=provider, model_name=model_name, api_key=api_key)
        
        # Run query (automatically applies state reset inside agent)
        result = ask_question(agent, question, df)
        
        if result.get("error"):
            print(f"[Query execution error]: {result['error']}", flush=True)
            return {
                "answer": result["answer"],
                "error": result["error"],
                "code": result.get("generated_code", "")
            }
            
        # Convert Matplotlib chart figure to base64 if it exists
        chart_base64 = None
        chart_fig = result.get("chart")
        if chart_fig is not None:
            try:
                buf = io.BytesIO()
                chart_fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
            except Exception as chart_err:
                print(f"Failed to serialize Matplotlib figure: {chart_err}")
                
        # Convert captured_df to JSON/dict
        captured_data = None
        captured_df = result.get("captured_df")
        if captured_df is not None:
            captured_data = captured_df.to_dict(orient="records")
            
        return {
            "answer": result["answer"],
            "code": result.get("generated_code", ""),
            "chart": chart_base64,
            "captured_df": captured_data,
            "error": None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
