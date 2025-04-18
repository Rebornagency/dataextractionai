"""
Enhanced API Server Module for NOI Analyzer
This module is updated to work with the new clearly labeled document approach
and properly handle document types from the NOI Tool
"""

import os
import logging
import json
import tempfile
import shutil
from typing import Dict, Any, List, Optional, Union
from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import local modules
from document_classifier import classify_document, map_noi_tool_to_extraction_type
from gpt_data_extractor import extract_financial_data
from preprocessing_module import preprocess_file
from validation_formatter import validate_and_format_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('noi_extraction_api')

# Get API key from environment variable
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    logger.warning("API_KEY environment variable not set. API will be accessible without authentication.")

# Create FastAPI app
app = FastAPI(
    title="NOI Data Extraction API",
    description="API for extracting financial data from real estate documents",
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

# Define request models
class BatchExtractionRequest(BaseModel):
    file_urls: List[str]
    document_types: Optional[Dict[str, str]] = None

class ExtractionResponse(BaseModel):
    document_type: str
    period: Optional[str]
    rental_income: float
    laundry_income: float
    parking_income: float
    other_revenue: float
    total_revenue: float
    repairs_maintenance: float
    utilities: float
    property_management_fees: float
    property_taxes: float
    insurance: float
    admin_office_costs: float
    marketing_advertising: float
    total_expenses: float
    net_operating_income: float

# Helper function to validate API key
def validate_api_key(api_key: Optional[str] = None, request: Optional[Request] = None) -> bool:
    """
    Validate API key from header or parameter
    
    Args:
        api_key: API key from parameter
        request: FastAPI request object
        
    Returns:
        True if API key is valid, False otherwise
    """
    # Log environment variable for debugging
    env_api_key = os.environ.get("API_KEY")
    logger.info(f"API_KEY environment variable is {'set' if env_api_key else 'NOT set'}")
    if env_api_key:
        logger.info(f"API_KEY length: {len(env_api_key)}")
    
    # If API_KEY is not set in environment, use a fallback key for development
    if not API_KEY:
        logger.warning("API_KEY environment variable not set. Using fallback authentication.")
        # Allow all requests in development mode
        return True
    
    # Log incoming API key information for debugging
    logger.info(f"Parameter API key is {'provided' if api_key else 'NOT provided'}")
    if request:
        logger.info(f"Headers: {list(request.headers.keys())}")
        logger.info(f"x-api-key header is {'present' if 'x-api-key' in request.headers else 'NOT present'}")
        logger.info(f"Authorization header is {'present' if 'authorization' in request.headers else 'NOT present'}")
    
    # Check API key from parameter
    if api_key:
        logger.info(f"Comparing parameter API key (length: {len(api_key)}) with environment API_KEY")
        if api_key == API_KEY:
            logger.info("API key validated via parameter")
            return True
    
    # Check API key from request headers
    if request:
        # Check x-api-key header (common API key header)
        header_api_key = request.headers.get('x-api-key')
        if header_api_key:
            logger.info(f"Comparing x-api-key header (length: {len(header_api_key)}) with environment API_KEY")
            if header_api_key == API_KEY:
                logger.info("API key validated via x-api-key header")
                return True
        
        # Check Authorization header (Bearer token)
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.replace('Bearer ', '')
            logger.info(f"Comparing Bearer token (length: {len(token)}) with environment API_KEY")
            if token == API_KEY:
                logger.info("API key validated via Bearer token")
                return True
    
    # For development purposes, allow requests with a specific test key
    test_key = "test-key-for-development"
    if api_key == test_key or (request and request.headers.get('x-api-key') == test_key):
        logger.warning("Using test key for development - NOT SECURE FOR PRODUCTION")
        return True
    
    # Log failure reason
    logger.warning("Invalid API key or missing API key")
    if request:
        logger.warning(f"Request headers: {dict(request.headers)}")
    return False

# Health check endpoint
@app.get("/health")
async def health_check(request: Request):
    """
    Health check endpoint
    """
    if not validate_api_key(request=request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    return {"status": "healthy", "version": "2.0.0"}

# Extract data from a single file
@app.post("/extract")
async def extract_data(
    file: UploadFile = File(...),
    document_type: Optional[str] = Form(None),
    api_key: Optional[str] = Header(None),
    request: Request = None
):
    """
    Extract financial data from a single file
    
    Args:
        file: Uploaded file
        document_type: Type of document (optional, from NOI Tool)
        api_key: API key for authentication (optional)
        request: FastAPI request object
        
    Returns:
        Extracted financial data
    """
    # Validate API key
    if not validate_api_key(api_key, request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        # Log request information
        logger.info(f"Processing file: {file.filename} (type: {file.content_type})")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Copy uploaded file to temporary file
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Preprocess file
            logger.info(f"Preprocessing file: {file.filename}")
            preprocessed_data = preprocess_file(temp_file_path, file.content_type, file.filename)
            logger.info(f"Preprocessed data type: {type(preprocessed_data)}")
            
            # Add filename to metadata if it's a dictionary
            if isinstance(preprocessed_data, dict) and 'metadata' not in preprocessed_data:
                preprocessed_data['metadata'] = {'filename': file.filename}
            elif isinstance(preprocessed_data, dict) and 'metadata' in preprocessed_data:
                preprocessed_data['metadata']['filename'] = file.filename
            
            # Classify document if document_type is not provided
            if document_type:
                # Map NOI Tool document type to extraction tool document type
                extraction_doc_type = map_noi_tool_to_extraction_type(document_type)
                logger.info(f"Using provided document type: {document_type} (mapped to: {extraction_doc_type})")
                
                # Extract period from filename or content
                period = None
                if hasattr(file, 'filename') and file.filename:
                    # Try to extract period from filename
                    import re
                    # Pattern for month and year (e.g., "March 2025", "Mar 2025", "Mar_2025")
                    month_year_pattern = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[_\s-]*\d{4}'
                    match = re.search(month_year_pattern, file.filename, re.IGNORECASE)
                    if match:
                        period = match.group(0).replace('_', ' ').strip()
                        logger.info(f"Extracted period from filename: {period}")
            else:
                # Classify document if document_type is not provided
                logger.info(f"Classifying document: {file.filename}")
                extraction_doc_type, period = classify_document(preprocessed_data)
            
            # Extract financial data
            logger.info(f"Extracting financial data: {file.filename}")
            extraction_result = extract_financial_data(preprocessed_data, extraction_doc_type, period)
            
            # Validate and format data
            logger.info(f"Validating and formatting data: {file.filename}")
            formatted_result = validate_and_format_data(extraction_result)
            
            logger.info(f"Successfully extracted data from {file.filename}")
            return formatted_result
            
        finally:
            # Remove temporary file
            os.unlink(temp_file_path)
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# Extract data from multiple files (batch processing)
@app.post("/extract-batch")
async def extract_batch(
    files: List[UploadFile] = File(...),
    document_types: Optional[str] = Form(None),
    api_key: Optional[str] = Header(None),
    request: Request = None
):
    """
    Extract financial data from multiple files
    
    Args:
        files: List of uploaded files
        document_types: JSON string mapping filenames to document types (optional)
        api_key: API key for authentication (optional)
        request: FastAPI request object
        
    Returns:
        List of extracted financial data
    """
    # Validate API key
    if not validate_api_key(api_key, request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Parse document_types if provided
    doc_types_map = {}
    if document_types:
        try:
            doc_types_map = json.loads(document_types)
        except json.JSONDecodeError:
            logger.warning(f"Invalid document_types JSON: {document_types}")
    
    results = []
    
    for file in files:
        try:
            # Get document type for this file if available
            doc_type = doc_types_map.get(file.filename)
            
            # Process file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                # Copy uploaded file to temporary file
                shutil.copyfileobj(file.file, temp_file)
                temp_file_path = temp_file.name
            
            try:
                # Preprocess file
                preprocessed_data = preprocess_file(temp_file_path, file.content_type, file.filename)
                
                # Add filename to metadata if it's a dictionary
                if isinstance(preprocessed_data, dict) and 'metadata' not in preprocessed_data:
                    preprocessed_data['metadata'] = {'filename': file.filename}
                elif isinstance(preprocessed_data, dict) and 'metadata' in preprocessed_data:
                    preprocessed_data['metadata']['filename'] = file.filename
                
                # Classify document if document_type is not provided
                if doc_type:
                    # Map NOI Tool document type to extraction tool document type
                    extraction_doc_type = map_noi_tool_to_extraction_type(doc_type)
                    
                    # Extract period from filename or content
                    period = None
                    if hasattr(file, 'filename') and file.filename:
                        # Try to extract period from filename
                        import re
                        # Pattern for month and year (e.g., "March 2025", "Mar 2025", "Mar_2025")
                        month_year_pattern = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[_\s-]*\d{4}'
                        match = re.search(month_year_pattern, file.filename, re.IGNORECASE)
                        if match:
                            period = match.group(0).replace('_', ' ').strip()
                else:
                    # Classify document if document_type is not provided
                    extraction_doc_type, period = classify_document(preprocessed_data)
                
                # Extract financial data
                extraction_result = extract_financial_data(preprocessed_data, extraction_doc_type, period)
                
                # Validate and format data
                formatted_result = validate_and_format_data(extraction_result)
                
                # Add filename to result
                formatted_result['filename'] = file.filename
                
                results.append(formatted_result)
                
            finally:
                # Remove temporary file
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            # Add error result
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return results

if __name__ == "__main__":
    # Run API server
    uvicorn.run(app, host="0.0.0.0", port=8000)
