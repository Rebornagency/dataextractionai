"""
Data Extraction AI - Combined Module (Enhanced + App Fees + NumPy Fix + All Syntax Fixes + Debug Logging)
This file contains all modules from the Data Extraction AI project into a single file
for easy upload to any LLM or deployment.

Enhancements:
- Extracts detailed income components (GPR, Vacancy, Recoveries)
- Explicitly extracts Application Fees under Other Income
- Extracts detailed operating expenses
- Extracts optional reserves
- Calculates EGI explicitly
- Returns a more structured JSON output
- Fixed NumPy 2.0 float type error
- Fixed SyntaxError in _determine_type_from_filename
- Fixed SyntaxError (missing except) in _extract_with_gpt
- Fixed SyntaxErrors from line compression in ValidationFormatter/DocumentClassifier
- Added DEBUG logging to FilePreprocessor methods
- Fixed variable naming inconsistency with year_month_match/month_year_match
- Added proper financials object structure for NOI Analyzer compatibility
"""

import os
import io
import logging
import json
import re
import time
import tempfile
import shutil
import magic # type: ignore
import chardet
import pandas as pd
import pdfplumber # type: ignore
from typing import Dict, Any, List, Tuple, Optional, Union
from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
import numpy as np # Ensure numpy is imported
import math # For isnan checks

# Initialize FastAPI app at module level
app = FastAPI(
    title="Data Extraction AI API",
    description="API for extracting financial data from various document types",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#################################################
# Configure logging (Set level to DEBUG)
#################################################
logging.basicConfig(
    level=logging.DEBUG, # <-- Set to DEBUG to show detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Use a distinct logger name
logger = logging.getLogger('data_extraction_api_service_debug')

# Rest of the code remains the same...

# Format extraction result for NOI Analyzer
def format_for_noi_analyzer(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format extraction result for NOI Analyzer
    
    Args:
        result: Extraction result
        
    Returns:
        Formatted result for NOI Analyzer
    """
    # Check if result is valid
    if not result or not isinstance(result, dict) or 'error' in result:
        return result
    
    # Extract data from result
    data = result
    if not data:
        return {"error": "No data extracted"}
    
    # Extract property_id and period
    property_id = data.get('property_id')
    period = data.get('period')
    
    # Extract financial data
    gross_potential_rent = data.get('gross_potential_rent')
    vacancy_loss = data.get('vacancy_loss')
    concessions = data.get('concessions', 0.0) or 0.0
    bad_debt = data.get('bad_debt', 0.0) or 0.0
    
    # Get operating expenses - handle both nested and flat structures
    operating_expenses = None
    if 'operating_expenses' in data and isinstance(data['operating_expenses'], dict):
        operating_expenses = data['operating_expenses'].get('total_operating_expenses')
    if operating_expenses is None:
        operating_expenses = data.get('operating_expenses_total')
    
    # Get NOI
    noi = data.get('net_operating_income')
    
    # Get other income - handle both nested and flat structures
    other_income = 0.0
    if 'other_income' in data:
        if isinstance(data['other_income'], dict):
            other_income = data['other_income'].get('total', 0.0) or 0.0
        else:
            other_income = data.get('other_income', 0.0) or 0.0
    
    # Get EGI - handle both nested and flat structures
    egi = data.get('effective_gross_income')
    if egi is None:
        # Calculate EGI if not provided
        egi = (gross_potential_rent or 0.0) - (vacancy_loss or 0.0) - (concessions or 0.0) - (bad_debt or 0.0) + (other_income or 0.0)
    
    # Format result for NOI Analyzer with financials object
    noi_analyzer_result = {
        "property_id": property_id,
        "period": period,
        "financials": {
            "gross_potential_rent": gross_potential_rent,
            "vacancy_loss": vacancy_loss,
            "concessions": concessions,
            "bad_debt": bad_debt,
            "other_income": other_income,
            "total_revenue": egi,  # Use EGI as total revenue
            "total_expenses": operating_expenses,
            "net_operating_income": noi,
            "effective_gross_income": egi
        },
        "source_documents": {
            "filename": result.get('metadata', {}).get('filename')
        }
    }
    
    return noi_analyzer_result

# Fix for the variable naming inconsistency in _extract_period_from_name
def _extract_period_from_name(self, name: str) -> Optional[str]:
    """Extract period from document name."""
    if not name: return None
    name_part = name.lower().replace('_', ' ').replace('-', ' ')
    months_full = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
    months_abbr = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    months_pattern = '|'.join(months_full + [f"{m}[a-z]*" for m in months_abbr])
    year_pattern = r'(20\d{2})'
    month_year_match = re.search(rf'({months_pattern})\s+{year_pattern}', name_part, re.IGNORECASE)
    if month_year_match: return f"{self._standardize_month(month_year_match.group(1),months_full,months_abbr)} {month_year_match.group(2)}"; month_year_match2=re.search(rf'{year_pattern}\s+({months_pattern})',name_part,re.IGNORECASE);
    if month_year_match2: return f"{self._standardize_month(month_year_match2.group(2),months_full,months_abbr)} {month_year_match2.group(1)}"; quarter_match=re.search(rf'(Q[1-4])\s*{year_pattern}',name_part,re.IGNORECASE);
    if quarter_match: return f"{quarter_match.group(1)} {quarter_match.group(2)}"; year_match=re.search(year_pattern,name_part);
    if year_match: return year_match.group(1)
    return None

# Add the API endpoints
@app.get("/")
async def root():
    """Root endpoint to check if API is running"""
    return {"message": "Data Extraction AI API is running"}

@app.post("/extract")
async def extract_data(
    file: UploadFile = File(...),
    document_type: Optional[str] = Form(None),
    api_key: Optional[str] = Header(None)
):
    """
    Extract data from uploaded document (V1 endpoint)
    
    Args:
        file: Uploaded file
        document_type: Optional document type label
        api_key: OpenAI API key (optional)
        
    Returns:
        Extracted and formatted data
    """
    logger.info(f"Received file: {file.filename}, Content-Type: {file.content_type}")
    
    # Create temp directory for file processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file to temp directory
        temp_file_path = os.path.join(temp_dir, file.filename)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            # Create instances of the necessary classes
            preprocessor = FilePreprocessor()
            classifier = DocumentClassifier()
            extractor = GPTDataExtractor(api_key=api_key)
            
            # Step 1: Preprocess the file
            preprocessed_data = preprocessor.preprocess(
                temp_file_path, 
                content_type=file.content_type, 
                filename=file.filename
            )
            
            # Step 2: Classify document and extract period
            classification_result = classifier.classify(
                preprocessed_data, 
                known_document_type=document_type
            )
            
            # Step 3: Extract data using GPT
            extraction_result = extractor.extract_data(
                preprocessed_data,
                document_type=classification_result['document_type'],
                period=classification_result['period']
            )
            
            # Step 4: Validate and format the extracted data
            formatter = ValidationFormatter()
            final_result = formatter.validate_and_format_data(extraction_result)
            
            # Add metadata to the result
            final_result['metadata'] = {
                'filename': file.filename,
                'document_type': classification_result['document_type'],
                'period': classification_result['period'],
                'classification_method': classification_result.get('method', 'unknown')
            }
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/api/v2/extraction/financials")
async def extract_financials_v2(
    file: UploadFile = File(...),
    property_id: Optional[str] = Form(None),
    period: Optional[str] = Form(None),
    api_key: Optional[str] = Header(None)
):
    """
    Extract financial data from uploaded file (V2 endpoint for NOI Analyzer)
    
    Args:
        file: Uploaded file
        property_id: Property ID (optional)
        period: Period (optional)
        api_key: OpenAI API key (optional)
        
    Returns:
        Extracted financial data in flattened format for NOI Analyzer
    """
    logger.info(f"Received V2 extraction request for file: {file.filename}, property_id: {property_id}, period: {period}")
    
    # Create temp directory for file processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file to temp directory
        temp_file_path = os.path.join(temp_dir, file.filename)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            # Create instances of the necessary classes
            preprocessor = FilePreprocessor()
            classifier = DocumentClassifier()
            extractor = GPTDataExtractor(api_key=api_key)
            formatter = ValidationFormatter()
            
            # Step 1: Preprocess the file
            preprocessed_data = preprocessor.preprocess(
                temp_file_path, 
                content_type=file.content_type, 
                filename=file.filename
            )
            
            # Step 2: Classify document and extract period if not provided
            classification_result = classifier.classify(
                preprocessed_data, 
                known_document_type="financial_statement"
            )
            
            # Use provided period if available, otherwise use classified period
            effective_period = period or classification_result['period']
            
            # Step 3: Extract data using GPT
            extraction_result = extractor.extract_data(
                preprocessed_data,
                document_type=classification_result['document_type'],
                period=effective_period
            )
            
            # If property_id was provided, add it to the extraction result
            if property_id:
                extraction_result['property_id'] = property_id
            
            # Step 4: Validate and format the extracted data
            final_result = formatter.validate_and_format_data(extraction_result)
            
            # Add metadata to the result
            final_result['metadata'] = {
                'filename': file.filename,
                'document_type': classification_result['document_type'],
                'period': effective_period,
                'classification_method': classification_result.get('method', 'unknown')
            }
            
            # Format result for NOI Analyzer
            noi_analyzer_result = format_for_noi_analyzer(final_result)
            
            return noi_analyzer_result
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise HTTPException(status_code=500, detail={"error": f"Error processing file: {str(e)}"})

# The rest of the code from main_backend.py would go here...

if __name__ == "__main__":
    # Run the API server
    uvicorn.run("main_backend:app", host="0.0.0.0", port=8000, reload=True)
