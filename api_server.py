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
from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import datetime
from openai import OpenAI, RateLimitError, APIError, APITimeoutError

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
# Document Classifier Class
#################################################
class DocumentClassifier:
    """
    Class for classifying financial documents
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.logger = logging.getLogger('document_classifier')
    
    def classify(self, preprocessed_data: Dict[str, Any], known_document_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Classify document and extract period
        
        Args:
            preprocessed_data: Preprocessed data from FilePreprocessor
            known_document_type: Optional known document type
            
        Returns:
            Dictionary with document type and period
        """
        # Extract text and filename
        text = preprocessed_data.get('text', '')
        filename = preprocessed_data.get('filename', '')
        
        # If document type is provided, use it
        if known_document_type:
            document_type = known_document_type
            method = "user_provided"
        else:
            # Try to determine from filename
            document_type = self._determine_type_from_filename(filename)
            if document_type:
                method = "filename"
            else:
                # Try to determine from content
                document_type = self._determine_type_from_content(text)
                method = "content"
        
        # Extract period from filename or content
        period = self._extract_period_from_name(filename)
        if not period:
            period = self._extract_period_from_content(text)
        
        return {
            'document_type': document_type or "financial_statement",
            'period': period,
            'method': method
        }
    
    def _determine_type_from_filename(self, filename: str) -> Optional[str]:
        """Determine document type from filename"""
        if not filename:
            return None
            
        filename_lower = filename.lower()
        
        # Check for budget indicators
        if 'budget' in filename_lower:
            return 'budget'
        
        # Check for actual/current indicators
        if any(term in filename_lower for term in ['actual', 'current']):
            return 'current_month_actuals'
        
        # Check for prior/previous indicators
        if any(term in filename_lower for term in ['prior', 'previous']):
            if 'year' in filename_lower:
                return 'prior_year_actuals'
            else:
                return 'prior_month_actuals'
        
        return None
    
    def _determine_type_from_content(self, text: str) -> str:
        """Determine document type from content"""
        text_lower = text.lower()
        
        # Check for budget indicators
        if any(term in text_lower for term in ['budget', 'projected', 'forecast']):
            return 'budget'
        
        # Check for actual indicators
        if any(term in text_lower for term in ['actual', 'ytd', 'year to date']):
            return 'current_month_actuals'
        
        # Default to financial statement
        return 'financial_statement'
    
    def _extract_period_from_name(self, name: str) -> Optional[str]:
        """Extract period from document name"""
        if not name:
            return None
            
        name_part = name.lower().replace('_', ' ').replace('-', ' ')
        
        # Define month patterns
        months_full = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
        months_abbr = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        months_pattern = '|'.join(months_full + [f"{m}[a-z]*" for m in months_abbr])
        
        # Define year pattern
        year_pattern = r'(20\d{2})'
        
        # Try to match Month Year pattern
        month_year_match = re.search(rf'({months_pattern})\s+{year_pattern}', name_part, re.IGNORECASE)
        if month_year_match:
            return f"{self._standardize_month(month_year_match.group(1), months_full, months_abbr)} {month_year_match.group(2)}"
        
        # Try to match Year Month pattern
        month_year_match2 = re.search(rf'{year_pattern}\s+({months_pattern})', name_part, re.IGNORECASE)
        if month_year_match2:
            return f"{self._standardize_month(month_year_match2.group(2), months_full, months_abbr)} {month_year_match2.group(1)}"
        
        # Try to match Quarter pattern
        quarter_match = re.search(rf'(Q[1-4])\s*{year_pattern}', name_part, re.IGNORECASE)
        if quarter_match:
            return f"{quarter_match.group(1)} {quarter_match.group(2)}"
        
        # Try to match just year
        year_match = re.search(year_pattern, name_part)
        if year_match:
            return year_match.group(1)
        
        return None
    
    def _extract_period_from_content(self, text: str) -> Optional[str]:
        """Extract period from document content"""
        if not text:
            return None
            
        text_lower = text.lower()
        
        # Define month patterns
        months_full = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
        months_abbr = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        months_pattern = '|'.join(months_full + [f"{m}[a-z]*" for m in months_abbr])
        
        # Define year pattern
        year_pattern = r'(20\d{2})'
        
        # Try to match Month Year pattern
        month_year_match = re.search(rf'({months_pattern})\s+{year_pattern}', text_lower, re.IGNORECASE)
        if month_year_match:
            return f"{self._standardize_month(month_year_match.group(1), months_full, months_abbr)} {month_year_match.group(2)}"
        
        # Try to match Year Month pattern
        month_year_match2 = re.search(rf'{year_pattern}\s+({months_pattern})', text_lower, re.IGNORECASE)
        if month_year_match2:
            return f"{self._standardize_month(month_year_match2.group(2), months_full, months_abbr)} {month_year_match2.group(1)}"
        
        # Try to match just year
        year_match = re.search(year_pattern, text_lower)
        if year_match:
            return year_match.group(1)
        
        return None
    
    def _standardize_month(self, month_str: str, months_full: List[str], months_abbr: List[str]) -> str:
        """Standardize month name"""
        month_str = month_str.lower()
        
        # Check full month names
        for i, month in enumerate(months_full):
            if month_str == month or month_str.startswith(month):
                return months_full[i].capitalize()
        
        # Check abbreviated month names
        for i, month in enumerate(months_abbr):
            if month_str == month or month_str.startswith(month):
                return months_full[i].capitalize()
        
        return month_str.capitalize()

#################################################
# GPT Data Extractor Class
#################################################
class GPTDataExtractor:
    """
    Class for extracting data using GPT
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.logger = logging.getLogger('gpt_data_extractor')
    
    def extract_data(self, preprocessed_data: Dict[str, Any], document_type: str, period: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract data from preprocessed data using GPT
        
        Args:
            preprocessed_data: Preprocessed data from FilePreprocessor
            document_type: Type of document
            period: Period of the document
            
        Returns:
            Extracted data
        """
        # Extract text and tables
        text = preprocessed_data.get('text', '')
        tables = preprocessed_data.get('tables', [])
        
        # Combine text and tables
        combined_text = text
        if tables:
            combined_text += "\n\n--- TABLES ---\n\n" + "\n\n".join(tables)
        
        # Truncate if too long
        if len(combined_text) > 15000:
            self.logger.warning(f"Text too long ({len(combined_text)} chars), truncating to 15000 chars")
            combined_text = combined_text[:15000]
        
        # Extract data using GPT
        try:
            extraction_result = self._extract_with_gpt(combined_text, document_type, period)
            return extraction_result
        except Exception as e:
            self.logger.error(f"Error extracting data with GPT: {str(e)}")
            return {'error': f"Error extracting data: {str(e)}"}
    
    def _extract_with_gpt(self, text: str, document_type: str, period: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract data using GPT
        
        Args:
            text: Text to extract from
            document_type: Type of document
            period: Period of the document
            
        Returns:
            Extracted data
        """
        # Create prompt
        prompt = self._create_extraction_prompt(text, document_type, period)
        
        try:
            # Initialize OpenAI client
            client = OpenAI(api_key=self.api_key)
            
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial data extraction assistant. Extract the requested financial data from the provided document."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=2000
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without markdown
                json_match = re.search(r'({.*})', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response_text
            
            # Parse JSON
            try:
                result = json.loads(json_str)
                
                # Add period if provided
                if period:
                    result['period'] = period
                
                return result
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing JSON from GPT response: {str(e)}")
                return {'error': f"Error parsing JSON from GPT response: {str(e)}"}
                
        except RateLimitError:
            self.logger.error("OpenAI API rate limit exceeded")
            return {'error': "OpenAI API rate limit exceeded. Please try again later."}
        except APIError as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            return {'error': f"OpenAI API error: {str(e)}"}
        except APITimeoutError:
            self.logger.error("OpenAI API timeout")
            return {'error': "OpenAI API timeout. Please try again later."}
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {str(e)}")
            return {'error': f"Error calling OpenAI API: {str(e)}"}
    
    def _create_extraction_prompt(self, text: str, document_type: str, period: Optional[str] = None) -> str:
        """
        Create extraction prompt for GPT
        
        Args:
            text: Text to extract from
            document_type: Type of document
            period: Period of the document
            
        Returns:
            Extraction prompt
        """
        prompt = f"""
        Extract financial data from the following {document_type} document.
        
        {f"Period: {period}" if period else ""}
        
        Document content:
        ```
        {text}
        ```
        
        Extract the following financial data in JSON format:
        
        1. Gross Potential Rent (GPR)
        2. Vacancy Loss
        3. Concessions (if available)
        4. Bad Debt (if available)
        5. Other Income, including:
           - Application Fees (if available)
           - Any additional income items with their names and amounts
           - Total Other Income
        6. Effective Gross Income (EGI)
        7. Operating Expenses, including:
           - Payroll
           - Administrative
           - Marketing
           - Utilities
           - Repairs and Maintenance
           - Contract Services
           - Make Ready
           - Turnover
           - Property Taxes
           - Insurance
           - Management Fees
           - Any other operating expenses with their names and amounts
           - Total Operating Expenses
        8. Net Operating Income (NOI)
        
        Format the response as a JSON object with the following structure:
        ```json
        {{
            "gross_potential_rent": <value>,
            "vacancy_loss": <value>,
            "concessions": <value>,
            "bad_debt": <value>,
            "other_income": {{
                "application_fees": <value>,
                "additional_items": [
                    {{ "name": "<item_name>", "amount": <value> }},
                    ...
                ],
                "total": <value>
            }},
            "effective_gross_income": <value>,
            "operating_expenses": {{
                "payroll": <value>,
                "administrative": <value>,
                "marketing": <value>,
                "utilities": <value>,
                "repairs_maintenance": <value>,
                "contract_services": <value>,
                "make_ready": <value>,
                "turnover": <value>,
                "property_taxes": <value>,
                "insurance": <value>,
                "management_fees": <value>,
                "other_operating_expenses": [
                    {{ "name": "<item_name>", "amount": <value> }},
                    ...
                ],
                "total_operating_expenses": <value>
            }},
            "net_operating_income": <value>
        }}
        ```
        
        Use null for missing values. All amounts should be numeric values without currency symbols or commas.
        """
        
        return prompt

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
        "version": "2.0.0",
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
            classifier = DocumentClassifier(api_key=api_key)
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
            # Create instances of the necessary classes
            preprocessor = FilePreprocessor()
            classifier = DocumentClassifier(api_key=api_key)
            extractor = GPTDataExtractor(api_key=api_key)
            
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

if __name__ == "__main__":
    # Run the API server
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
