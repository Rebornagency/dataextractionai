import os
import logging
from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import shutil
from typing import Dict, Any, Optional, List

from preprocessing_module import preprocess_file
from document_classifier import DocumentClassifier
from gpt_data_extractor import GPTDataExtractor
from validation_formatter import validate_and_format_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('api_server')

# Initialize FastAPI app
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
            # Step 1: Preprocess the file
            preprocessed_data = preprocess_file(
                temp_file_path, 
                content_type=file.content_type, 
                filename=file.filename
            )
            
            # Step 2: Classify document and extract period
            classifier = DocumentClassifier(api_key=api_key)
            classification_result = classifier.classify(
                preprocessed_data, 
                known_document_type=document_type
            )
            
            # Step 3: Extract data using GPT
            extractor = GPTDataExtractor(api_key=api_key)
            extraction_result = extractor.extract_data(
                preprocessed_data,
                document_type=classification_result['document_type'],
                period=classification_result['period']
            )
            
            # Step 4: Validate and format the extracted data
            final_result = validate_and_format_data(extraction_result)
            
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
            # Step 1: Preprocess the file
            preprocessed_data = preprocess_file(
                temp_file_path, 
                content_type=file.content_type, 
                filename=file.filename
            )
            
            # Step 2: Classify document and extract period if not provided
            classifier = DocumentClassifier(api_key=api_key)
            classification_result = classifier.classify(
                preprocessed_data, 
                known_document_type="financial_statement"
            )
            
            # Use provided period if available, otherwise use classified period
            effective_period = period or classification_result['period']
            
            # Step 3: Extract data using GPT
            extractor = GPTDataExtractor(api_key=api_key)
            extraction_result = extractor.extract_data(
                preprocessed_data,
                document_type=classification_result['document_type'],
                period=effective_period
            )
            
            # If property_id was provided, add it to the extraction result
            if property_id:
                extraction_result['property_id'] = property_id
            
            # Step 4: Validate and format the extracted data
            final_result = validate_and_format_data(extraction_result)
            
            # Add metadata to the result
            final_result['metadata'] = {
                'filename': file.filename,
                'document_type': classification_result['document_type'],
                'period': effective_period,
                'classification_method': classification_result.get('method', 'unknown')
            }
            
            # Format result for NOI Analyzer
            noi_analyzer_result = format_for_noi_analyzer(final_result)
            
            # Validate required fields
            missing_fields = validate_required_fields(noi_analyzer_result)
            if missing_fields:
                raise HTTPException(
                    status_code=422,
                    detail={"error": f"Missing fields: {', '.join(missing_fields)}"}
                )
            
            return noi_analyzer_result
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise HTTPException(status_code=500, detail={"error": f"Error processing file: {str(e)}"})

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

def validate_required_fields(data: Dict[str, Any]) -> List[str]:
    """
    Validate required fields for NOI Analyzer
    
    Args:
        data: Data to validate
        
    Returns:
        List of missing required fields
    """
    required_fields = [
        'property_id',
        'period',
        'financials'
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)
    
    # Also check required fields in financials
    if 'financials' in data and data['financials'] is not None:
        financials_required_fields = [
            'gross_potential_rent',
            'vacancy_loss',
            'total_expenses',
            'net_operating_income'
        ]
        
        for field in financials_required_fields:
            if field not in data['financials'] or data['financials'][field] is None:
                missing_fields.append(f"financials.{field}")
    
    return missing_fields

if __name__ == "__main__":
    # Run the API server
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
