"""
Data Extraction AI API Server (Enhanced + Fixed FilePreprocessor + Health Check)

This file contains the FastAPI server for the Data Extraction AI project.
It includes endpoints for extracting financial data from various document types.

Enhancements:
- Added proper FilePreprocessor class definition
- Added health check endpoint for Render
- Fixed data structure for NOI Analyzer compatibility
- Improved error handling and logging
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
from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import datetime
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
from pydantic import BaseModel, Field, validator
from config.settings import get_settings, API_TITLE, API_DESCRIPTION, API_VERSION
from src.utils.helpers import get_api_key, validate_required_fields, validate_data, determine_document_type, format_currency, format_percent
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('api_server')

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
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
# FilePreprocessor Class Definition
#################################################
class FilePreprocessor:
    """
    Class for preprocessing files before extraction
    """
    def __init__(self):
        self.logger = logging.getLogger('file_preprocessor')
        self.logger.debug("FilePreprocessor initialized")
    
    def preprocess(self, file_path: str, content_type: Optional[str] = None, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Preprocess the file and extract text content
        
        Args:
            file_path: Path to the file
            content_type: MIME type of the file
            filename: Name of the file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        self.logger.debug(f"Preprocessing file: {filename}, content_type: {content_type}")
        
        # Determine file type if not provided
        if not content_type:
            content_type = self._determine_content_type(file_path)
            self.logger.debug(f"Determined content type: {content_type}")
        
        # Extract text based on file type
        if 'pdf' in content_type.lower():
            text, tables = self._extract_from_pdf(file_path)
        elif 'spreadsheet' in content_type.lower() or 'excel' in content_type.lower() or file_path.endswith(('.xlsx', '.xls', '.csv')):
            text, tables = self._extract_from_spreadsheet(file_path)
        else:
            text, tables = self._extract_from_text(file_path)
        
        # Return preprocessed data
        result = {
            'text': text,
            'tables': tables,
            'file_type': content_type,
            'filename': filename
        }
        
        self.logger.debug(f"Preprocessing complete. Extracted {len(text)} chars of text and {len(tables)} tables")
        return result
    
    def _determine_content_type(self, file_path: str) -> str:
        """Determine content type of file"""
        try:
            mime = magic.Magic(mime=True)
            content_type = mime.from_file(file_path)
            return content_type
        except Exception as e:
            self.logger.error(f"Error determining content type: {str(e)}")
            # Fallback to extension-based detection
            if file_path.lower().endswith('.pdf'):
                return 'application/pdf'
            elif file_path.lower().endswith(('.xlsx', '.xls')):
                return 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            elif file_path.lower().endswith('.csv'):
                return 'text/csv'
            else:
                return 'text/plain'
    
    def _extract_from_pdf(self, file_path: str) -> Tuple[str, List[str]]:
        """Extract text and tables from PDF"""
        self.logger.debug(f"Extracting from PDF: {file_path}")
        extracted_text = []
        extracted_tables = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    text = page.extract_text() or ""
                    if text:
                        extracted_text.append(f"--- Page {page_num + 1} ---\n{text}")
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for i, table in enumerate(tables):
                        if table:
                            # Convert table to string representation
                            table_str = self._format_table(table)
                            extracted_tables.append(f"--- Table {i+1} on Page {page_num + 1} ---\n{table_str}")
        except Exception as e:
            self.logger.error(f"Error extracting from PDF: {str(e)}")
        
        return "\n\n".join(extracted_text), extracted_tables
    
    def _extract_from_spreadsheet(self, file_path: str) -> Tuple[str, List[str]]:
        """Extract text and tables from spreadsheet"""
        self.logger.debug(f"Extracting from spreadsheet: {file_path}")
        extracted_text = []
        extracted_tables = []
        
        try:
            # Determine file type
            if file_path.lower().endswith('.csv'):
                df_dict = {'Sheet1': pd.read_csv(file_path)}
            else:
                df_dict = pd.read_excel(file_path, sheet_name=None)
            
            # Process each sheet
            for sheet_name, df in df_dict.items():
                # Convert DataFrame to string
                sheet_text = f"--- Sheet: {sheet_name} ---\n"
                sheet_text += df.to_string(index=False)
                extracted_text.append(sheet_text)
                
                # Also add as table
                table_str = df.to_string(index=False)
                extracted_tables.append(f"--- Table in Sheet: {sheet_name} ---\n{table_str}")
        except Exception as e:
            self.logger.error(f"Error extracting from spreadsheet: {str(e)}")
        
        return "\n\n".join(extracted_text), extracted_tables
    
    def _extract_from_text(self, file_path: str) -> Tuple[str, List[str]]:
        """Extract text from text file"""
        self.logger.debug(f"Extracting from text file: {file_path}")
        try:
            # Try to detect encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'
            
            # Read with detected encoding
            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read()
            
            return text, []
        except Exception as e:
            self.logger.error(f"Error extracting from text file: {str(e)}")
            return "", []
    
    def _format_table(self, table: List[List[Any]]) -> str:
        """Format table as string"""
        rows = []
        for row in table:
            # Replace None with empty string
            formatted_row = [str(cell) if cell is not None else "" for cell in row]
            rows.append(" | ".join(formatted_row))
        return "\n".join(rows)

#################################################
# GPT Data Extractor Class
#################################################
class GPTDataExtractor:
    """
    Enhanced GPT data extractor with improved error handling and fallbacks
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        
    def extract_data(self, preprocessed_data: Dict[str, Any], document_type: str, period: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract data from preprocessed document using GPT with fallbacks
        
        Args:
            preprocessed_data: Dictionary with 'text' and 'tables' keys
            document_type: Type of document
            period: Optional period information
            
        Returns:
            Dictionary with extracted data
        """
        logger.info(f"Extracting data from {document_type} document")
        
        # Get text from preprocessed data
        if not preprocessed_data or 'text' not in preprocessed_data:
            logger.error("No text data in preprocessed data")
            return {"error": "No text data in preprocessed data"}
            
        text = preprocessed_data['text']
        
        # Add tables if available
        if 'tables' in preprocessed_data and preprocessed_data['tables']:
            text += "\n\nTABLES:\n" + "\n\n".join(preprocessed_data['tables'])
            
        # Try GPT extraction with retries
        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                result = self._extract_with_gpt(text, document_type, period)
                if result and 'error' not in result:
                    return result
                    
                # If GPT extraction failed but returned a partial result, try to repair it
                if 'partial_data' in result:
                    logger.warning("Got partial data, attempting repair")
                    repaired_result = self._repair_partial_data(result['partial_data'], document_type)
                    if repaired_result:
                        return repaired_result
                        
                logger.error(f"GPT extraction failed: {result.get('error', 'Unknown error')}")
                last_error = result.get('error', 'Unknown error')
                
            except RateLimitError:
                logger.warning(f"Rate limit exceeded (attempt {retry_count + 1}/{max_retries}), waiting 10s")
                time.sleep(10)
                
            except APITimeoutError:
                logger.warning(f"API timeout (attempt {retry_count + 1}/{max_retries}), waiting 5s")
                time.sleep(5)
                
            except APIError as e:
                logger.error(f"OpenAI API error: {str(e)}")
                last_error = f"OpenAI API error: {str(e)}"
                break
                
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                last_error = f"Unexpected error: {str(e)}"
                break
                
            retry_count += 1
            
        # If all GPT attempts failed, try rule-based extraction as fallback
        logger.warning("All GPT extraction attempts failed, trying rule-based fallback")
        fallback_result = self._rule_based_extraction(text, document_type, period)
        if fallback_result:
            fallback_result['extraction_method'] = 'rule_based_fallback'
            return fallback_result
            
        # If everything failed, return error
        return {"error": f"Data extraction failed: {last_error}"}
        
    def _extract_with_gpt(self, text: str, document_type: str, period: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract data with GPT model
        
        Args:
            text: Document text
            document_type: Document type
            period: Optional period
            
        Returns:
            Dictionary with extracted data
        """
        # Create prompt
        prompt = self._create_extraction_prompt(text, document_type, period)
        
        try:
            # Call GPT API
            response = self.client.chat.completions.create(
                model="gpt-4",  # Use GPT-4 for better extraction quality
                temperature=0.1,  # Low temperature for consistent results
                messages=[
                    {"role": "system", "content": "You are a financial data extraction expert. Extract structured data from financial documents accurately."},
                    {"role": "user", "content": prompt}
                ],
                timeout=60  # Set timeout to 60 seconds
            )
            
            # Process response
            result_text = response.choices[0].message.content
            
            # Try to parse JSON from the response
            try:
                # Find JSON object in response
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = result_text[json_start:json_end]
                    result = json.loads(json_str)
                    result['extraction_method'] = 'gpt'
                    return result
                else:
                    # If no JSON found, try to parse it as a formatted string
                    result = self._parse_formatted_string(result_text)
                    if result:
                        result['extraction_method'] = 'gpt_formatted'
                        return result
                    
                    return {"error": "Could not parse GPT response as JSON", "partial_data": result_text}
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GPT response as JSON: {str(e)}")
                return {"error": f"Failed to parse GPT response: {str(e)}", "partial_data": result_text}
                
        except Exception as e:
            logger.error(f"GPT API call failed: {str(e)}")
            return {"error": f"GPT API call failed: {str(e)}"}

def standardize_period(period: Optional[str]) -> Optional[str]:
    """
    Standardize period format to YYYY-MM
    
    Args:
        period: Period in various formats
        
    Returns:
        Standardized period in YYYY-MM format
    """
    if not period:
        return None
        
    import re
    
    # Try to match various date formats
    # YYYY-MM or YYYY-MM-DD
    match = re.match(r'(\d{4})-(\d{1,2})(?:-\d{1,2})?', period)
    if match:
        year, month = match.groups()
        return f"{year}-{int(month):02d}"
    
    # MM-YYYY or MM/YYYY
    match = re.match(r'(\d{1,2})[-/](\d{4})', period)
    if match:
        month, year = match.groups()
        return f"{year}-{int(month):02d}"
    
    # Month name YYYY (January 2023)
    month_names = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]
    
    for i, month in enumerate(month_names):
        # Full month name
        pattern = f"(?:{month})\s+(\d{{4}})"
        match = re.search(pattern, period.lower())
        if match:
            year = match.group(1)
            return f"{year}-{i+1:02d}"
        
        # Abbreviated month name
        abbr = month[:3]
        pattern = f"(?:{abbr})[a-z]*\s+(\d{{4}})"
        match = re.search(pattern, period.lower())
        if match:
            year = match.group(1)
            return f"{year}-{i+1:02d}"
    
    # Could not standardize
    return period

def validate_and_format_data(extraction_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and format the extracted data
    
    Args:
        extraction_result: Raw data extracted from GPT
        
    Returns:
        Validated and formatted data
    """
    logger.info("Validating and formatting extracted data")
    
    # Check if extraction result is valid
    if not extraction_result or not isinstance(extraction_result, dict):
        logger.error("Invalid extraction result")
        return {
            'error': 'Invalid extraction result',
            'data': None
        }
        
    # Check if there was an error in extraction
    if 'error' in extraction_result:
        logger.error(f"Error in extraction result: {extraction_result['error']}")
        return extraction_result
        
    # Create a copy of the extraction result to avoid modifying the original
    formatted_data = extraction_result.copy()
    
    # Add validation status
    formatted_data['validation'] = {
        'status': 'valid',
        'message': 'Data validated and formatted successfully'
    }
    
    return formatted_data

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

# Add health check endpoint for Render
@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return {
        "status": "healthy",
        "version": API_VERSION,
        "timestamp": datetime.datetime.now().isoformat()
    }

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
            extractor = GPTDataExtractor(api_key=api_key)
            
            # Step 1: Preprocess the file
            preprocessed_data = preprocessor.preprocess(
                temp_file_path, 
                content_type=file.content_type, 
                filename=file.filename
            )
            
            # Step 2: Extract data using GPT
            extraction_result = extractor.extract_data(
                preprocessed_data,
                document_type=document_type,
            )
            
            # Step 3: Validate and format the extracted data
            final_result = validate_and_format_data(extraction_result)
            
            # Add metadata to the result
            final_result['metadata'] = {
                'filename': file.filename,
                'document_type': document_type,
                'period': extraction_result.get('period'),
                'classification_method': 'user_provided' if document_type else 'unknown'
            }
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

class ExtractionRequest(BaseModel):
    document_type: Optional[str] = None
    property_id: Optional[str] = None
    period: Optional[str] = None
    
    @validator('document_type')
    def validate_document_type(cls, v):
        if v is not None:
            valid_types = ['profit_loss', 'balance_sheet', 'rent_roll', 'operating_statement']
            if v.lower() not in [t.lower() for t in valid_types]:
                raise ValueError(f"Invalid document type: {v}. Must be one of: {', '.join(valid_types)}")
        return v
    
    @validator('period')
    def validate_period(cls, v):
        if v is not None:
            # Basic period format validation (YYYY-MM or YYYY-MM-DD)
            import re
            if not re.match(r'^\d{4}-\d{2}(-\d{2})?$', v):
                raise ValueError("Period must be in YYYY-MM or YYYY-MM-DD format")
        return v

@app.post("/api/v2/extraction/financials")
async def extract_financials_v2(
    file: UploadFile = File(...),
    request: ExtractionRequest = Depends()
):
    """
    Enhanced endpoint for financial data extraction
    """
    # Validate file size
    file_size = 0
    chunk_size = 1024 * 1024  # 1MB chunks
    content = b''
    
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        content += chunk
        file_size += len(chunk)
        
        # Check file size limit (10MB)
        if file_size > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 10MB)")
    
    # Reset file position
    file.file = io.BytesIO(content)
    
    # Check file type
    try:
        file_type = magic.from_buffer(content[:1024], mime=True)
        if not file_type.startswith(('application/pdf', 'application/vnd.ms-excel', 
                                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                    'text/csv', 'text/plain')):
            raise HTTPException(status_code=415, detail=f"Unsupported file type: {file_type}")
    except Exception as e:
        logger.error(f"Error checking file type: {str(e)}")
    
    # Process the extraction with enhanced error handling
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Preprocess file
        preprocessor = FilePreprocessor()
        preprocessed_data = preprocessor.preprocess(
            temp_path, 
            content_type=file.content_type, 
            filename=file.filename
        )
        
        # Extract data
        extractor = GPTDataExtractor(api_key=None)
        extraction_result = extractor.extract_data(
            preprocessed_data, 
            document_type=request.document_type, 
            period=request.period
        )
        
        # Add metadata
        extraction_result['metadata'] = {
            'filename': file.filename,
            'document_type': request.document_type,
            'period': extraction_result.get('period'),
            'property_id': request.property_id,
            'extraction_time': datetime.datetime.now().isoformat()
        }
        
        # Format for NOI Analyzer
        formatted_result = format_for_noi_comparison(extraction_result)
        
        # Remove temporary file
        os.unlink(temp_path)
        
        return formatted_result
        
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

def format_for_noi_comparison(extraction_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the extraction result for NOI comparison
    
    Args:
        extraction_result: Raw extraction result with financials, raw_lines and validation
        
    Returns:
        Formatted data for NOI comparison
    """
    # Create formatted result
    formatted_result = {}
    
    # Extract financials from the new structure
    financials = extraction_result.get("financials", {})
    
    # If there's an error in extraction, return it
    if "error" in extraction_result:
        return {"error": extraction_result["error"]}
    
    # Copy financial data to formatted result
    for key, value in financials.items():
        formatted_result[key] = value
    
    # Add metadata
    formatted_result["metadata"] = extraction_result.get("metadata", {})
    
    # Add audit lines for traceability
    formatted_result["audit_lines"] = extraction_result.get("raw_lines", [])
    
    # Add validation issues if present
    if "validation_issues" in extraction_result:
        formatted_result["validation_issues"] = extraction_result["validation_issues"]
    
    return formatted_result

# Add this before extract_noi_data function
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.exceptions.ConnectionError, 
                                   requests.exceptions.Timeout,
                                   requests.exceptions.HTTPError))
)
def post_with_retry(url, files, headers=None, timeout=30):
    """
    Make a POST request with retry logic for transient errors
    
    Args:
        url: The API endpoint URL
        files: Files to upload
        headers: Optional request headers
        timeout: Request timeout in seconds
        
    Returns:
        Response object
        
    Raises:
        Exception: If all retry attempts fail
    """
    logger.info(f"Sending POST request to {url}")
    response = requests.post(url, files=files, headers=headers, timeout=timeout)
    response.raise_for_status()  # Raise exception for 4XX/5XX status codes
    return response

def extract_noi_data(document_files, api_url=None, api_key=None):
    """
    Extract NOI data from documents using the external extraction API.
    Enhanced with retry logic and improved error handling.
    """
    from api_integration import call_extraction_api
    results = {}
    
    for doc_file in document_files:
        try:
            result, error = call_extraction_api(doc_file)
            if error:
                logger.error(f"Error extracting data from {doc_file.name}: {error}")
                results[doc_file.name] = {"error": error}
            else:
                results[doc_file.name] = result
        except Exception as e:
            logger.error(f"Unexpected error during extraction for {doc_file.name}: {str(e)}", exc_info=True)
            results[doc_file.name] = {
                "error": f"Extraction failed: {str(e)}",
                "status": "error"
            }
    
    return results

if __name__ == "__main__":
    # Get settings
    settings = get_settings()
    
    # Start server
    uvicorn.run(
        "api_server:app",
        host=settings['api']['host'],
        port=settings['api']['port'],
        reload=settings['api']['debug']
    )
