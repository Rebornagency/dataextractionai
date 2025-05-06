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
- Fixed FilePreprocessor not defined error
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
import datetime

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

#################################################
# Validation Formatter Class
#################################################
class ValidationFormatter:
    """
    Class for validating and formatting extracted data
    """
    def __init__(self):
        self.logger = logging.getLogger('validation_formatter')
    
    def validate_and_format_data(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and format the extracted data
        
        Args:
            extraction_result: Raw data extracted from GPT
            
        Returns:
            Validated and formatted data
        """
        self.logger.info("Validating and formatting extracted data")
        
        # Check if extraction result is valid
        if not extraction_result or not isinstance(extraction_result, dict):
            self.logger.error("Invalid extraction result")
            return {
                'error': 'Invalid extraction result',
                'data': None
            }
            
        # Check if there was an error in extraction
        if 'error' in extraction_result:
            self.logger.error(f"Error in extraction result: {extraction_result['error']}")
            return extraction_result
            
        # Create a copy of the extraction result to avoid modifying the original
        formatted_data = extraction_result.copy()
        
        try:
            # Validate and format income data
            formatted_data = self._validate_income_fields(formatted_data)
                
            # Validate and format operating expenses
            formatted_data = self._validate_operating_expenses_fields(formatted_data)
                
            # Validate and format reserves
            formatted_data = self._validate_reserves_fields(formatted_data)
                
            # Validate and format NOI
            formatted_data['net_operating_income'] = self._validate_noi(
                formatted_data.get('net_operating_income'),
                formatted_data.get('effective_gross_income'),
                formatted_data.get('operating_expenses', {}).get('total_operating_expenses')
            )
            
            # Validate property_id and period
            formatted_data['property_id'] = formatted_data.get('property_id')
            formatted_data['period'] = self._validate_period(formatted_data.get('period'))
            
            # Perform additional validation checks
            formatted_data = self._perform_validation_checks(formatted_data)
            
            # Add validation status
            formatted_data['validation'] = {
                'status': 'valid',
                'message': 'Data validated and formatted successfully'
            }
            
        except Exception as e:
            self.logger.error(f"Error validating and formatting data: {str(e)}")
            formatted_data['validation'] = {
                'status': 'error',
                'message': f"Error validating and formatting data: {str(e)}"
            }
            
        return formatted_data
    
    def _validate_income_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and format income fields
        
        Args:
            data: Raw data
            
        Returns:
            Validated and formatted data
        """
        # Extract income fields from flattened structure
        gross_potential_rent = self._validate_numeric(data.get('gross_potential_rent'))
        vacancy_loss = self._validate_numeric(data.get('vacancy_loss'))
        concessions = self._validate_numeric(data.get('concessions'))
        bad_debt = self._validate_numeric(data.get('bad_debt'))
        recoveries = self._validate_numeric(data.get('recoveries'))
        
        # Handle other_income which is still a nested structure
        other_income = data.get('other_income', {})
        if not isinstance(other_income, dict):
            other_income = {}
        
        # Validate standard other income categories
        parking = self._validate_numeric(other_income.get('parking'))
        laundry = self._validate_numeric(other_income.get('laundry'))
        late_fees = self._validate_numeric(other_income.get('late_fees'))
        pet_fees = self._validate_numeric(other_income.get('pet_fees'))
        application_fees = self._validate_numeric(other_income.get('application_fees'))
        
        # Validate additional items
        additional_items = []
        if 'additional_items' in other_income and isinstance(other_income['additional_items'], list):
            for item in other_income['additional_items']:
                if isinstance(item, dict) and 'name' in item and 'amount' in item:
                    validated_item = {
                        'name': str(item['name']),
                        'amount': self._validate_numeric(item['amount'])
                    }
                    additional_items.append(validated_item)
        
        other_income_total = self._validate_numeric(other_income.get('total'))
        
        # Calculate other_income total if not provided
        calculated_other_income = 0
        for value in [parking, laundry, late_fees, pet_fees, application_fees]:
            if value is not None:
                calculated_other_income += value
                
        # Add additional items to the total
        for item in additional_items:
            if item['amount'] is not None:
                calculated_other_income += item['amount']
        
        if calculated_other_income > 0:
            if other_income_total is None:
                other_income_total = calculated_other_income
            elif not self._is_close(other_income_total, calculated_other_income, 0.05):
                self.logger.warning(f"Provided other income total ({other_income_total}) differs from calculated total ({calculated_other_income})")
                other_income_total = calculated_other_income
        
        # Validate EGI
        effective_gross_income = self._validate_numeric(data.get('effective_gross_income'))
        
        # Calculate EGI if not provided
        calculated_egi = self._calculate_egi(
            gross_potential_rent,
            vacancy_loss,
            concessions,
            bad_debt,
            recoveries,
            other_income_total
        )
        
        if calculated_egi is not None:
            if effective_gross_income is None:
                effective_gross_income = calculated_egi
            elif not self._is_close(effective_gross_income, calculated_egi, 0.05):
                self.logger.warning(f"Provided EGI ({effective_gross_income}) differs from calculated EGI ({calculated_egi})")
                effective_gross_income = calculated_egi
        
        # Update data with validated values
        data['gross_potential_rent'] = gross_potential_rent
        data['vacancy_loss'] = vacancy_loss
        data['concessions'] = concessions
        data['bad_debt'] = bad_debt
        data['recoveries'] = recoveries
        data['other_income'] = {
            'parking': parking,
            'laundry': laundry,
            'late_fees': late_fees,
            'pet_fees': pet_fees,
            'application_fees': application_fees,
            'additional_items': additional_items,
            'total': other_income_total
        }
        data['effective_gross_income'] = effective_gross_income
        
        # Add individual other income components to top level for NOI Analyzer
        data['parking'] = parking
        data['laundry'] = laundry
        data['late_fees'] = late_fees
        data['pet_fees'] = pet_fees
        data['application_fees'] = application_fees
        
        return data
    
    def _validate_operating_expenses_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and format operating expenses fields
        
        Args:
            data: Raw data
            
        Returns:
            Validated and formatted data
        """
        # Extract operating expenses from nested structure
        operating_expenses = data.get('operating_expenses', {})
        if not isinstance(operating_expenses, dict):
            operating_expenses = {}
        
        # Validate standard expense categories
        payroll = self._validate_numeric(operating_expenses.get('payroll'))
        administrative = self._validate_numeric(operating_expenses.get('administrative'))
        marketing = self._validate_numeric(operating_expenses.get('marketing'))
        utilities = self._validate_numeric(operating_expenses.get('utilities'))
        repairs_maintenance = self._validate_numeric(operating_expenses.get('repairs_maintenance'))
        contract_services = self._validate_numeric(operating_expenses.get('contract_services'))
        make_ready = self._validate_numeric(operating_expenses.get('make_ready'))
        turnover = self._validate_numeric(operating_expenses.get('turnover'))
        property_taxes = self._validate_numeric(operating_expenses.get('property_taxes'))
        insurance = self._validate_numeric(operating_expenses.get('insurance'))
        management_fees = self._validate_numeric(operating_expenses.get('management_fees'))
        
        # Validate other operating expenses
        other_operating_expenses = []
        if 'other_operating_expenses' in operating_expenses and isinstance(operating_expenses['other_operating_expenses'], list):
            for item in operating_expenses['other_operating_expenses']:
                if isinstance(item, dict) and 'name' in item and 'amount' in item:
                    validated_item = {
                        'name': str(item['name']),
                        'amount': self._validate_numeric(item['amount'])
                    }
                    other_operating_expenses.append(validated_item)
        
        # Validate total operating expenses
        total_operating_expenses = self._validate_numeric(operating_expenses.get('total_operating_expenses'))
        
        # Calculate total if not provided
        calculated_total = None
        for value in [payroll, administrative, marketing, utilities, repairs_maintenance, 
                     contract_services, make_ready, turnover, property_taxes, insurance, management_fees]:
            if value is not None:
                if calculated_total is None:
                    calculated_total = value
                else:
                    calculated_total += value
        
        # Add other operating expenses to the total
        for item in other_operating_expenses:
            if item['amount'] is not None:
                if calculated_total is None:
                    calculated_total = item['amount']
                else:
                    calculated_total += item['amount']
        
        if calculated_total is not None:
            if total_operating_expenses is None:
                total_operating_expenses = calculated_total
            elif not self._is_close(total_operating_expenses, calculated_total, 0.05):
                self.logger.warning(f"Provided operating expenses total ({total_operating_expenses}) differs from calculated total ({calculated_total})")
                total_operating_expenses = calculated_total
        
        # Update data with validated values
        data['operating_expenses'] = {
            'payroll': payroll,
            'administrative': administrative,
            'marketing': marketing,
            'utilities': utilities,
            'repairs_maintenance': repairs_maintenance,
            'contract_services': contract_services,
            'make_ready': make_ready,
            'turnover': turnover,
            'property_taxes': property_taxes,
            'insurance': insurance,
            'management_fees': management_fees,
            'other_operating_expenses': other_operating_expenses,
            'total_operating_expenses': total_operating_expenses
        }
        
        # Also add total_operating_expenses to top level for NOI Analyzer
        data['operating_expenses_total'] = total_operating_expenses
        
        # Add individual operating expense components to top level for NOI Analyzer
        data['property_taxes'] = property_taxes
        data['insurance'] = insurance
        data['repairs_and_maintenance'] = repairs_maintenance  # Note the normalized name
        data['utilities'] = utilities
        data['management_fees'] = management_fees
        
        return data
    
    def _validate_reserves_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and format reserves fields
        
        Args:
            data: Raw data
            
        Returns:
            Validated and formatted data
        """
        # Extract reserves from nested structure
        reserves = data.get('reserves', {})
        if not isinstance(reserves, dict):
            reserves = {}
        
        # Validate standard reserve categories
        replacement_reserves = self._validate_numeric(reserves.get('replacement_reserves'))
        capital_expenditures = self._validate_numeric(reserves.get('capital_expenditures'))
        
        # Validate other reserves
        other_reserves = []
        if 'other_reserves' in reserves and isinstance(reserves['other_reserves'], list):
            for item in reserves['other_reserves']:
                if isinstance(item, dict) and 'name' in item and 'amount' in item:
                    validated_item = {
                        'name': str(item['name']),
                        'amount': self._validate_numeric(item['amount'])
                    }
                    other_reserves.append(validated_item)
        
        # Validate total reserves
        total_reserves = self._validate_numeric(reserves.get('total_reserves'))
        
        # Calculate total if not provided
        calculated_total = None
        if replacement_reserves is not None:
            calculated_total = replacement_reserves
        
        if capital_expenditures is not None:
            if calculated_total is None:
                calculated_total = capital_expenditures
            else:
                calculated_total += capital_expenditures
        
        # Add other reserves to the total
        for item in other_reserves:
            if item['amount'] is not None:
                if calculated_total is None:
                    calculated_total = item['amount']
                else:
                    calculated_total += item['amount']
        
        if calculated_total is not None:
            if total_reserves is None:
                total_reserves = calculated_total
            elif not self._is_close(total_reserves, calculated_total, 0.05):
                self.logger.warning(f"Provided reserves total ({total_reserves}) differs from calculated total ({calculated_total})")
                total_reserves = calculated_total
        
        # Update data with validated values
        data['reserves'] = {
            'replacement_reserves': replacement_reserves,
            'capital_expenditures': capital_expenditures,
            'other_reserves': other_reserves,
            'total_reserves': total_reserves
        }
        
        return data
    
    def _perform_validation_checks(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform additional validation checks
        
        Args:
            data: Data to validate
            
        Returns:
            Validated data
        """
        # 1. Non-negative Values Check
        for field in ['gross_potential_rent', 'vacancy_loss', 'concessions', 'bad_debt', 'operating_expenses_total', 'net_operating_income']:
            value = data.get(field)
            if value is not None and value < 0:
                self.logger.warning(f"Negative value found for {field}: {value}. Setting to 0.")
                data[field] = 0.0
        
        # 2. EGI Sanity Check
        gpr = data.get('gross_potential_rent', 0) or 0
        vacancy_loss = data.get('vacancy_loss', 0) or 0
        concessions = data.get('concessions', 0) or 0
        bad_debt = data.get('bad_debt', 0) or 0
        other_income_total = data.get('other_income', {}).get('total', 0) or 0
        
        calculated_egi = gpr - (vacancy_loss + concessions + bad_debt) + other_income_total
        
        if calculated_egi < 0:
            self.logger.warning(f"Calculated EGI is negative: {calculated_egi}. Adjusting vacancy/concessions/bad_debt.")
            # Reset problematic values to 0
            data['vacancy_loss'] = 0.0
            data['concessions'] = 0.0
            data['bad_debt'] = 0.0
            # Recalculate EGI
            data['effective_gross_income'] = gpr + other_income_total
        
        # 3. NOI Consistency Check
        egi = data.get('effective_gross_income', 0) or 0
        opex = data.get('operating_expenses_total', 0) or 0
        noi = data.get('net_operating_income', 0) or 0
        
        calculated_noi = egi - opex
        
        if noi is not None and calculated_noi is not None:
            if not self._is_close(noi, calculated_noi, 0.01):
                self.logger.warning(f"NOI consistency check failed: provided NOI ({noi}) differs from calculated NOI ({calculated_noi})")
                data['net_operating_income'] = calculated_noi
        
        # Set fallback values for non-required fields
        if data.get('concessions') is None:
            data['concessions'] = 0.0
        
        if data.get('bad_debt') is None:
            data['bad_debt'] = 0.0
        
        return data
    
    def _validate_period(self, period: Optional[str]) -> Optional[str]:
        """
        Validate and format period string to YYYY-MM format
        
        Args:
            period: Period string
            
        Returns:
            Validated period string
        """
        if not period:
            return None
        
        # Try to extract YYYY-MM format
        import re
        
        # Check if already in YYYY-MM format
        if re.match(r'^\d{4}-\d{2}$', period):
            return period
        
        # Try to extract year and month from various formats
        year_month_match = re.search(r'(\d{4})[-/\s]*(\d{1,2})', period)
        if year_month_match:
            year = year_month_match.group(1)
            month = year_month_match.group(2).zfill(2)  # Ensure month is 2 digits
            return f"{year}-{month}"
        
        # Try to extract from month name formats (e.g., "January 2025")
        month_names = {
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05', 'jun': '06',
            'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
        }
        
        month_year_match = re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,.-]+(\d{4})', 
                                    period.lower())
        if month_year_match:
            month_abbr = month_year_match.group(1)
            year = month_year_match.group(2)
            month = month_names.get(month_abbr, '01')
            return f"{year}-{month}"
        
        # If only year is found, use January as default month
        year_match = re.search(r'(\d{4})', period)
        if year_match:
            year = year_match.group(1)
            return f"{year}-01"
        
        return None
    
    def _validate_numeric(self, value: Any) -> Optional[float]:
        """
        Validate and convert to numeric value
        
        Args:
            value: Value to validate
            
        Returns:
            Validated numeric value
        """
        if value is None:
            return None
        
        # Handle numpy types
        if hasattr(value, 'item'):
            try:
                value = value.item()
            except:
                pass
        
        # Handle string values
        if isinstance(value, str):
            # Remove currency symbols and commas
            value = value.replace('$', '').replace(',', '').strip()
            
            # Handle empty string
            if not value:
                return None
            
            # Handle percentage values
            if value.endswith('%'):
                try:
                    return float(value.rstrip('%')) / 100
                except ValueError:
                    return None
            
            # Handle other string values
            try:
                return float(value)
            except ValueError:
                return None
        
        # Handle numeric values
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _is_close(self, a: Optional[float], b: Optional[float], rel_tol: float) -> bool:
        """
        Check if two values are close
        
        Args:
            a: First value
            b: Second value
            rel_tol: Relative tolerance
            
        Returns:
            True if values are close, False otherwise
        """
        if a is None or b is None:
            return False
        
        # Handle zero values
        if a == 0 and b == 0:
            return True
        
        # Handle very small values
        if abs(a) < 0.01 and abs(b) < 0.01:
            return True
        
        # Calculate relative difference
        rel_diff = abs(a - b) / max(abs(a), abs(b))
        
        return rel_diff <= rel_tol
    
    def _calculate_egi(
        self,
        gross_potential_rent: Optional[float],
        vacancy_loss: Optional[float],
        concessions: Optional[float],
        bad_debt: Optional[float],
        recoveries: Optional[float],
        other_income: Optional[float]
    ) -> Optional[float]:
        """
        Calculate Effective Gross Income
        
        Args:
            gross_potential_rent: Gross Potential Rent
            vacancy_loss: Vacancy Loss
            concessions: Concessions
            bad_debt: Bad Debt
            recoveries: Recoveries
            other_income: Other Income
            
        Returns:
            Calculated EGI
        """
        # If GPR is not provided, can't calculate EGI
        if gross_potential_rent is None:
            return None
        
        # Initialize with GPR
        egi = gross_potential_rent
        
        # Subtract vacancy loss
        if vacancy_loss is not None:
            egi -= vacancy_loss
        
        # Subtract concessions
        if concessions is not None:
            egi -= concessions
        
        # Subtract bad debt
        if bad_debt is not None:
            egi -= bad_debt
        
        # Add recoveries
        if recoveries is not None:
            egi += recoveries
        
        # Add other income
        if other_income is not None:
            egi += other_income
        
        return egi
    
    def _validate_noi(
        self,
        noi: Optional[float],
        egi: Optional[float],
        opex: Optional[float]
    ) -> Optional[float]:
        """
        Validate Net Operating Income
        
        Args:
            noi: Provided NOI
            egi: Effective Gross Income
            opex: Operating Expenses
            
        Returns:
            Validated NOI
        """
        # If EGI or OpEx is not provided, can't validate NOI
        if egi is None or opex is None:
            return noi
        
        # Calculate NOI
        calculated_noi = egi - opex
        
        # If NOI is not provided, use calculated NOI
        if noi is None:
            return calculated_noi
        
        # If NOI is provided but differs significantly from calculated NOI, use calculated NOI
        if not self._is_close(noi, calculated_noi, 0.05):
            self.logger.warning(f"Provided NOI ({noi}) differs from calculated NOI ({calculated_noi})")
            return calculated_noi
        
        return noi

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
    operating_expenses_components = {}
    
    if 'operating_expenses' in data and isinstance(data['operating_expenses'], dict):
        opex_data = data['operating_expenses']
        operating_expenses = opex_data.get('total_operating_expenses')
        
        # Extract OpEx components
        component_mapping = {
            "property_taxes": ["property_taxes", "taxes"],
            "insurance": ["insurance"],
            "repairs_and_maintenance": ["repairs_maintenance", "repairs", "maintenance"],
            "utilities": ["utilities"],
            "management_fees": ["management_fees", "management"]
        }
        
        # Try to extract each component using various possible field names
        for normalized_key, possible_fields in component_mapping.items():
            for field in possible_fields:
                if field in opex_data and opex_data[field] is not None:
                    operating_expenses_components[normalized_key] = opex_data[field]
                    break
    
    # Fall back to flat structure if needed
    if operating_expenses is None:
        operating_expenses = data.get('operating_expenses_total')
    
    # Get individual OpEx components from top level if not already found
    top_level_components = [
        ("property_taxes", "property_taxes"),
        ("insurance", "insurance"),
        ("repairs_and_maintenance", "repairs_maintenance"),
        ("utilities", "utilities"),
        ("management_fees", "management_fees")
    ]
    
    for component_key, data_key in top_level_components:
        if component_key not in operating_expenses_components and data_key in data:
            operating_expenses_components[component_key] = data.get(data_key)
    
    # Get other income - handle both nested and flat structures
    other_income = 0.0
    other_income_components = {}
    
    if 'other_income' in data:
        if isinstance(data['other_income'], dict):
            other_income_data = data['other_income']
            other_income = other_income_data.get('total', 0.0) or 0.0
            
            # Extract Other Income components
            component_mapping = {
                "parking": ["parking"],
                "laundry": ["laundry"],
                "late_fees": ["late_fees", "late fees"],
                "pet_fees": ["pet_fees", "pet fees", "pet rent"],
                "application_fees": ["application_fees", "application fees"]
            }
            
            # Try to extract each component using various possible field names
            for normalized_key, possible_fields in component_mapping.items():
                for field in possible_fields:
                    if field in other_income_data and other_income_data[field] is not None:
                        other_income_components[normalized_key] = other_income_data[field]
                        break
        else:
            other_income = data.get('other_income', 0.0) or 0.0
    
    # Get individual Other Income components from top level if not already found
    top_level_components = [
        ("parking", "parking"),
        ("laundry", "laundry"),
        ("late_fees", "late_fees"),
        ("pet_fees", "pet_fees"),
        ("application_fees", "application_fees")
    ]
    
    for component_key, data_key in top_level_components:
        if component_key not in other_income_components and data_key in data:
            other_income_components[component_key] = data.get(data_key)
    
    # Default missing categories to zero if we have a nonzero total
    if other_income and other_income > 0:
        for component_key in ["parking", "laundry", "late_fees", "pet_fees", "application_fees"]:
            if component_key not in other_income_components:
                other_income_components[component_key] = 0.0
    
    # Get NOI
    noi = data.get('net_operating_income')
    
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
            "effective_gross_income": egi,
            # Add OpEx component breakdowns
            "property_taxes": operating_expenses_components.get("property_taxes", 0.0),
            "insurance": operating_expenses_components.get("insurance", 0.0),
            "repairs_and_maintenance": operating_expenses_components.get("repairs_and_maintenance", 0.0),
            "utilities": operating_expenses_components.get("utilities", 0.0),
            "management_fees": operating_expenses_components.get("management_fees", 0.0),
            # Add Other Income component breakdowns
            "parking": other_income_components.get("parking", 0.0),
            "laundry": other_income_components.get("laundry", 0.0),
            "late_fees": other_income_components.get("late_fees", 0.0),
            "pet_fees": other_income_components.get("pet_fees", 0.0),
            "application_fees": other_income_components.get("application_fees", 0.0)
        },
        "source_documents": {
            "filename": result.get('metadata', {}).get('filename')
        }
    }
    
    return noi_analyzer_result

# Add health check endpoint for Render
@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.datetime.now().isoformat()
    }

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
            classifier = DocumentClassifier(api_key=api_key)
            extractor = GPTDataExtractor(api_key=api_key)
            formatter = ValidationFormatter()
            
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
            classifier = DocumentClassifier(api_key=api_key)
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

if __name__ == "__main__":
    # Run the API server
    uvicorn.run("main_backend:app", host="0.0.0.0", port=8000, reload=True)
