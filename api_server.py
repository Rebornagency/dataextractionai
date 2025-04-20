import os
import logging
from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import shutil
from typing import Dict, Any, Optional

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
    version="1.0.0"
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
    Extract data from uploaded document
    
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

if __name__ == "__main__":
    # Run the API server
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
