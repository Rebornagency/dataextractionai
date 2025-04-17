"""
Enhanced API Server with Support for Labeled Document Types
This server provides endpoints for extracting financial data from documents,
with support for the new clearly labeled document approach in the NOI Tool
"""

import os
import tempfile
import logging
from typing import List, Dict, Any, Optional, Union
import json

from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Form, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import openai

# Import preprocessing and extraction modules
from preprocessing_module import preprocess_file
from document_classifier import classify_document, extract_period_from_filename, map_noi_tool_to_extraction_type
from gpt_data_extractor import extract_financial_data
from validation_formatter import validate_and_format_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('noi_extraction_api')

# Initialize FastAPI app
app = FastAPI(
    title="NOI Data Extraction API",
    description="API for extracting financial data from real estate documents",
    version="1.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API key from environment variable
API_KEY = os.environ.get("API_KEY", "your-api-key")

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    logger.warning("OPENAI_API_KEY not set. GPT extraction will not work.")

# Response models
class ExtractedDataModel(BaseModel):
    document_type: str
    period: str
    financials: Dict[str, Any]

class BatchExtractedDataModel(BaseModel):
    results: List[ExtractedDataModel]
    consolidated_data: Dict[str, Any]

# Helper function to validate API key from different header formats
def validate_api_key(request: Request, api_key: Optional[str] = None):
    """
    Validate API key from different header formats for compatibility
    
    Args:
        request: FastAPI request object
        api_key: API key from standard Header parameter
        
    Returns:
        True if valid, raises HTTPException if invalid
    """
    # Log all headers for debugging
    logger.info(f"Request headers: {dict(request.headers)}")
    
    # Check standard header parameter first
    if api_key and api_key == API_KEY:
        logger.info("API key validated via standard header parameter")
        return True
    
    # Check x-api-key header (used by NOI tool)
    x_api_key = request.headers.get("x-api-key")
    if x_api_key and x_api_key == API_KEY:
        logger.info("API key validated via x-api-key header")
        return True
    
    # Check Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header == f"Bearer {API_KEY}":
        logger.info("API key validated via Authorization header")
        return True
    
    # If we get here, no valid API key was found
    logger.warning("Invalid API key or missing API key")
    raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/")
async def root():
    """
    Root endpoint for API health check
    """
    return {"status": "ok", "message": "NOI Data Extraction API is running"}

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "version": "1.1.0"}

@app.post("/extract", response_model=ExtractedDataModel)
async def extract_data(
    request: Request,
    file: UploadFile = File(...), 
    api_key: Optional[str] = Header(None),
    property_name: Optional[str] = None,
    document_type: Optional[str] = Query(None, description="Known document type from NOI Tool")
):
    """
    Extract financial data from a document
    
    Args:
        request: FastAPI request object
        file: Uploaded file (PDF, Excel, CSV, TXT)
        api_key: API key for authentication
        property_name: Optional property name
        document_type: Known document type from NOI Tool (optional)
        
    Returns:
        Extracted financial data
    """
    # Validate API key with compatibility for different header formats
    validate_api_key(request, api_key)
    
    # Get file extension
    file_ext = file.filename.split('.')[-1].lower()
    logger.info(f"Processing file: {file.filename} (type: {file_ext})")
    
    # Log document type if provided
    if document_type:
        logger.info(f"Known document type provided: {document_type}")
    
    # Save uploaded file to temp directory
    file_content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name
    
    try:
        # Step 1: Preprocess the file
        logger.info(f"Preprocessing file: {file.filename}")
        preprocessed_data = preprocess_file(temp_file_path)
        
        # Log the type of preprocessed_data for debugging
        logger.info(f"Preprocessed data type: {type(preprocessed_data)}")
        
        # Step 2: Classify the document, using known document type if provided
        logger.info(f"Classifying document: {file.filename}")
        
        # If document_type is provided, map it to extraction tool format
        known_extraction_type = None
        if document_type:
            known_extraction_type = document_type
            logger.info(f"Using known document type: {known_extraction_type}")
        
        doc_type, period = classify_document(preprocessed_data, known_extraction_type)
        
        # If period not found in content, try to extract from filename
        if not period:
            period = extract_period_from_filename(file.filename)
            logger.info(f"Extracted period from filename: {period}")
        
        # Step 3: Extract financial data
        logger.info(f"Extracting financial data: {file.filename}")
        extracted_data = extract_financial_data(preprocessed_data, doc_type, period)
        
        # Step 4: Validate and format data
        logger.info(f"Validating and formatting data: {file.filename}")
        validated_data = validate_and_format_data(extracted_data)
        
        # Ensure document_type and period are never null
        doc_type = doc_type or "Actual"  # Default to Actual if not determined
        period = period or "Unknown Period"
        
        # Return extracted data
        result = {
            "document_type": doc_type,
            "period": period,
            "financials": validated_data
        }
        
        logger.info(f"Successfully extracted data from {file.filename}")
        return result
    
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        # Include traceback for better debugging
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_file_path)
        except:
            pass

@app.post("/extract-batch", response_model=BatchExtractedDataModel)
async def extract_batch_data(
    request: Request,
    files: List[UploadFile] = File(...), 
    api_key: Optional[str] = Header(None),
    document_types: Optional[List[str]] = Query(None, description="Known document types from NOI Tool")
):
    """
    Extract financial data from multiple documents
    
    Args:
        request: FastAPI request object
        files: List of uploaded files (PDF, Excel, CSV, TXT)
        api_key: API key for authentication
        document_types: List of known document types from NOI Tool (optional)
        
    Returns:
        List of extracted financial data and consolidated data structure
    """
    # Validate API key with compatibility for different header formats
    validate_api_key(request, api_key)
    
    logger.info(f"Processing batch of {len(files)} files")
    
    # Log file names for debugging
    file_names = [file.filename for file in files]
    logger.info(f"Files to process: {file_names}")
    
    # Log document types if provided
    if document_types:
        logger.info(f"Known document types provided: {document_types}")
        # Ensure document_types matches the number of files
        if len(document_types) != len(files):
            logger.warning(f"Number of document types ({len(document_types)}) doesn't match number of files ({len(files)})")
    
    results = []
    for i, file in enumerate(files):
        # Get file extension
        file_ext = file.filename.split('.')[-1].lower()
        logger.info(f"Processing file: {file.filename} (type: {file_ext})")
        
        # Get known document type for this file if available
        known_doc_type = None
        if document_types and i < len(document_types):
            known_doc_type = document_types[i]
            logger.info(f"Using known document type for {file.filename}: {known_doc_type}")
        
        # Save uploaded file to temp directory
        file_content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Step 1: Preprocess the file
            logger.info(f"Preprocessing file: {file.filename}")
            preprocessed_data = preprocess_file(temp_file_path)
            
            # Log the type of preprocessed_data for debugging
            logger.info(f"Preprocessed data type: {type(preprocessed_data)}")
            
            # Step 2: Classify the document, using known document type if provided
            logger.info(f"Classifying document: {file.filename}")
            doc_type, period = classify_document(preprocessed_data, known_doc_type)
            
            # If period not found in content, try to extract from filename
            if not period:
                period = extract_period_from_filename(file.filename)
                logger.info(f"Extracted period from filename: {period}")
            
            # Step 3: Extract financial data
            logger.info(f"Extracting financial data: {file.filename}")
            extracted_data = extract_financial_data(preprocessed_data, doc_type, period)
            
            # Step 4: Validate and format data
            logger.info(f"Validating and formatting data: {file.filename}")
            validated_data = validate_and_format_data(extracted_data)
            
            # Ensure document_type and period are never null
            doc_type = doc_type or "Actual"  # Default to Actual if not determined
            period = period or "Unknown Period"
            
            # Add to results
            result = {
                "document_type": doc_type,
                "period": period,
                "financials": validated_data,
                "filename": file.filename,  # Include filename for reference
                "noi_tool_type": known_doc_type  # Include original NOI Tool document type if provided
            }
            results.append(result)
            
            logger.info(f"Successfully extracted data from {file.filename}")
        
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            # Include traceback for better debugging
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Add error result
            error_result = {
                "document_type": "Error",
                "period": "Error",
                "financials": {},
                "filename": file.filename,
                "error": str(e),
                "noi_tool_type": known_doc_type
            }
            results.append(error_result)
            
            # Continue processing other files even if one fails
            continue
        
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except:
                pass
    
    # If no files were successfully processed, return error
    if not results:
        raise HTTPException(status_code=500, detail="Failed to process any files")
    
    # Consolidate data for NOI tool
    consolidated_data = consolidate_data(results)
    
    # Log the final response structure for debugging
    response = {
        "results": results,
        "consolidated_data": consolidated_data
    }
    logger.info(f"Final response structure: {json.dumps(response, default=str)[:500]}...")
    
    return response

def consolidate_data(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Consolidate data from multiple documents into a structure for NOI comparisons,
    with priority given to known NOI Tool document types
    
    Args:
        results: List of extracted data from documents
        
    Returns:
        Consolidated data structure
    """
    logger.info("Consolidating data from multiple documents")
    
    # Initialize consolidated data structure
    consolidated = {
        "current_month": None,
        "prior_month": None,
        "budget": None,
        "prior_year": None
    }
    
    # First pass: Use known NOI Tool document types
    for result in results:
        # Skip errors
        if result["document_type"] == "Error":
            continue
        
        # If we have a known NOI Tool document type, use it directly
        noi_tool_type = result.get("noi_tool_type")
        if noi_tool_type:
            # Map the financials to the expected format
            formatted_data = {
                "revenue": result["financials"].get("total_revenue", 0),
                "expenses": result["financials"].get("total_expenses", 0),
                "noi": result["financials"].get("net_operating_income", 0)
            }
            
            # Assign to the appropriate slot
            if noi_tool_type == "current_month_actuals" and not consolidated["current_month"]:
                consolidated["current_month"] = formatted_data
                logger.info(f"Assigned {result['filename']} to current_month based on known type")
            elif noi_tool_type == "prior_month_actuals" and not consolidated["prior_month"]:
                consolidated["prior_month"] = formatted_data
                logger.info(f"Assigned {result['filename']} to prior_month based on known type")
            elif noi_tool_type == "current_month_budget" and not consolidated["budget"]:
                consolidated["budget"] = formatted_data
                logger.info(f"Assigned {result['filename']} to budget based on known type")
            elif noi_tool_type == "prior_year_actuals" and not consolidated["prior_year"]:
                consolidated["prior_year"] = formatted_data
                logger.info(f"Assigned {result['filename']} to prior_year based on known type")
    
    # Second pass: Fill in any missing slots using extracted document types
    for result in results:
        # Skip errors
        if result["document_type"] == "Error":
            continue
        
        # If this result already has a known NOI Tool type that was used, skip it
        if result.get("noi_tool_type") and any(
            consolidated[slot] is not None and 
            consolidated[slot]["revenue"] == result["financials"].get("total_revenue", 0) and
            consolidated[slot]["expenses"] == result["financials"].get("total_expenses", 0) and
            consolidated[slot]["noi"] == result["financials"].get("net_operating_income", 0)
            for slot in ["current_month", "prior_month", "budget", "prior_year"]
        ):
            continue
        
        # Format the financials
        formatted_data = {
            "revenue": result["financials"].get("total_revenue", 0),
            "expenses": result["financials"].get("total_expenses", 0),
            "noi": result["financials"].get("net_operating_income", 0)
        }
        
        # Assign based on document type
        doc_type = result["document_type"]
        if doc_type == "Actual Income Statement":
            # If we don't have a current month yet, use this one
            if not consolidated["current_month"]:
                consolidated["current_month"] = formatted_data
                logger.info(f"Assigned {result['filename']} to current_month based on document type")
            # If we already have a current month, use this as prior month
            elif not consolidated["prior_month"]:
                consolidated["prior_month"] = formatted_data
                logger.info(f"Assigned {result['filename']} to prior_month based on document type")
        elif doc_type == "Budget":
            if not consolidated["budget"]:
                consolidated["budget"] = formatted_data
                logger.info(f"Assigned {result['filename']} to budget based on document type")
        elif doc_type == "Prior Year Actual":
            if not consolidated["prior_year"]:
                consolidated["prior_year"] = formatted_data
                logger.info(f"Assigned {result['filename']} to prior_year based on document type")
    
    # If we have multiple Actual Income Statements but no explicit categorization,
    # try to determine which is current and which is prior based on dates
    if len([r for r in results if r["document_type"] == "Actual Income Statement"]) > 1:
        actuals = [r for r in results if r["document_type"] == "Actual Income Statement"]
        # Sort by period (assuming more recent is current)
        actuals.sort(key=lambda x: x["period"], reverse=True)
        
        # Assign current and prior
        if len(actuals) >= 1 and not consolidated["current_month"]:
            formatted_data = {
                "revenue": actuals[0]["financials"].get("total_revenue", 0),
                "expenses": actuals[0]["financials"].get("total_expenses", 0),
                "noi": actuals[0]["financials"].get("net_operating_income", 0)
            }
            consolidated["current_month"] = formatted_data
            logger.info(f"Assigned {actuals[0]['filename']} to current_month based on date sorting")
            
        if len(actuals) >= 2 and not consolidated["prior_month"]:
            formatted_data = {
                "revenue": actuals[1]["financials"].get("total_revenue", 0),
                "expenses": actuals[1]["financials"].get("total_expenses", 0),
                "noi": actuals[1]["financials"].get("net_operating_income", 0)
            }
            consolidated["prior_month"] = formatted_data
            logger.info(f"Assigned {actuals[1]['filename']} to prior_month based on date sorting")
    
    logger.info(f"Consolidated data: {json.dumps(consolidated, default=str)[:500]}...")
    return consolidated

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
