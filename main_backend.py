"""
Data Extraction AI - Combined Module (Enhanced + Application Fees)
This file contains all modules from the Data Extraction AI project into a single file
for easy upload to any LLM or deployment.

Original modules:
- api_server.py: FastAPI server for handling API requests
- document_classifier.py: Classifies financial documents and extracts time periods
- gpt_data_extractor.py: Extracts financial data using GPT-4
- preprocessing_module.py: Handles different file types (PDF, Excel, CSV, TXT)
- validation_formatter.py: Validates and formats the extracted data

Enhancements:
- Extracts detailed income components (GPR, Vacancy, Recoveries)
- Explicitly extracts Application Fees under Other Income
- Extracts detailed operating expenses
- Extracts optional reserves
- Calculates EGI explicitly
- Returns a more structured JSON output

Dependencies (install via requirements.txt):
- fastapi, uvicorn, openai, pandas, openpyxl, pdfplumber, python-magic,
- chardet, pydantic, python-dotenv, requests, python-multipart, numpy
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

#################################################
# Configure logging
#################################################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Use a distinct logger name
logger = logging.getLogger('data_extraction_api_service')

#################################################
# Preprocessing Module
#################################################
class FilePreprocessor:
    """Main class for preprocessing different file types"""

    def __init__(self):
        """Initialize the preprocessor"""
        self.supported_extensions = {
            'pdf': self._process_pdf,
            'xlsx': self._process_excel,
            'xls': self._process_excel,
            'csv': self._process_csv,
            'txt': self._process_txt
        }

    def preprocess(self, file_path: str, content_type: str = None, filename: str = None) -> Dict[str, Any]:
        """
        Main method to preprocess a file.

        Args:
            file_path: Path to the file to preprocess.
            content_type: Content type of the file (optional).
            filename: Original filename (optional).

        Returns:
            Dict containing extracted text/data and metadata.

        Raises:
            FileNotFoundError: If the file_path does not exist.
            ValueError: If the file type is unsupported.
            Exception: For underlying processing errors.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine file extension
        effective_filename = filename or file_path
        _, ext = os.path.splitext(effective_filename)
        ext = ext.lower().lstrip('.')

        # Determine file type from content_type if provided and reliable
        detected_type_from_content = None
        if content_type:
            content_type_map = {
                'application/pdf': 'pdf',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
                'application/vnd.ms-excel': 'xls',
                'text/csv': 'csv',
                'text/plain': 'txt'
            }
            normalized_content_type = content_type.split(';')[0].strip().lower()
            detected_type_from_content = content_type_map.get(normalized_content_type)

        # Use content type detection if available, otherwise use extension
        final_ext = detected_type_from_content or ext
        logger.info(f"Processing file '{effective_filename}'. Detected type: {final_ext} (from {'content-type' if detected_type_from_content else 'extension'}).")


        # Select processor based on final extension
        processor = self.supported_extensions.get(final_ext)

        # Fallback for generic spreadsheet types if extension wasn't xls/xlsx
        if not processor and content_type and ('spreadsheet' in content_type or 'excel' in content_type.lower()):
            logger.warning(f"Extension '{final_ext}' not directly supported, but content-type '{content_type}' suggests Excel. Attempting Excel processing.")
            processor = self._process_excel
        elif not processor:
             # Use python-magic as a final check before failing
             try:
                 mime_type = magic.from_file(file_path, mime=True)
                 logger.info(f"python-magic detected MIME type: {mime_type}")
                 # Add more robust mapping if needed based on magic types
                 if 'pdf' in mime_type: processor = self._process_pdf
                 elif 'excel' in mime_type or 'spreadsheet' in mime_type: processor = self._process_excel
                 elif 'csv' in mime_type: processor = self._process_csv
                 elif 'text' in mime_type: processor = self._process_txt
                 else: raise ValueError(f"Unsupported file type: {final_ext} (MIME: {mime_type})")
             except Exception as magic_err:
                 logger.error(f"python-magic check failed: {magic_err}")
                 raise ValueError(f"Unsupported file type: {final_ext}")


        # Get file metadata
        try:
            file_type_magic = magic.from_file(file_path, mime=True)
        except Exception as magic_err:
            logger.warning(f"python-magic failed to get MIME type: {magic_err}, using content_type if available.")
            file_type_magic = content_type or "unknown"
        file_size = os.path.getsize(file_path)

        logger.info(f"Processing with {processor.__name__}. File info: Type='{file_type_magic}', Size={file_size} bytes.")

        # Extract content using the selected processor
        try:
            extracted_content = processor(file_path)
        except Exception as proc_err:
            logger.error(f"Error during processing with {processor.__name__}: {proc_err}")
            raise # Re-raise the exception to be caught by the API endpoint

        # Assemble result
        result = {
            'metadata': {
                'filename': filename or os.path.basename(file_path),
                'file_type_source': file_type_magic, # From magic or content_type
                'file_size_bytes': file_size,
                'detected_extension': final_ext,
                'original_content_type': content_type
            },
            'content': extracted_content # Contains 'combined_text' and potentially other structures
        }

        return result

    def _process_pdf(self, file_path: str) -> Dict[str, Any]:
        """Process PDF files using pdfplumber."""
        logger.info(f"Extracting content from PDF: {os.path.basename(file_path)}")
        result = {'text': [], 'tables': []}
        all_text_content = []
        try:
            with pdfplumber.open(file_path) as pdf:
                logger.info(f"PDF has {len(pdf.pages)} pages.")
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    page_text = page.extract_text(x_tolerance=1, y_tolerance=3) # Adjust tolerances if needed
                    if page_text:
                        cleaned_page_text = self._clean_text(page_text)
                        result['text'].append({'page': page_num, 'content': cleaned_page_text})
                        all_text_content.append(cleaned_page_text)

                    # Extract tables with settings for better layout detection
                    try:
                        tables = page.extract_tables(table_settings={
                            "vertical_strategy": "lines",
                            "horizontal_strategy": "lines",
                            "snap_tolerance": 5, # Increased tolerance
                            "join_tolerance": 5, # Increased tolerance
                        })
                        for j, table in enumerate(tables):
                            if table:
                                # Clean table data (handle None, convert to string)
                                cleaned_table = [[str(cell) if cell is not None else '' for cell in row] for row in table]
                                if not cleaned_table: continue # Skip empty tables

                                df = pd.DataFrame(cleaned_table)
                                # Try to identify header row more robustly
                                header_index = self._find_header_row(df)
                                if header_index is not None:
                                    df.columns = df.iloc[header_index]
                                    df = df.iloc[header_index + 1:].reset_index(drop=True)
                                else:
                                    logger.debug(f"No clear header found in table {j+1} on page {page_num}, using default headers.")
                                    # Optionally generate generic headers: df.columns = [f'Col_{k}' for k in range(len(df.columns))]

                                # Convert NaN to None for JSON compatibility
                                df = df.replace({np.nan: None})
                                result['tables'].append({
                                    'page': page_num,
                                    'table_index': j,
                                    'data': df.to_dict(orient='records')
                                })
                    except Exception as table_err:
                        logger.warning(f"Could not extract tables from page {page_num}: {table_err}")

        except Exception as e:
            logger.error(f"Error processing PDF file '{os.path.basename(file_path)}': {str(e)}")
            raise # Re-raise to indicate processing failure

        result['combined_text'] = "\n\n--- Page Break ---\n\n".join(all_text_content)
        logger.info(f"Finished PDF processing. Extracted {len(result['text'])} text pages, {len(result['tables'])} tables. Total text length: {len(result['combined_text'])} chars.")
        return result

    def _process_excel(self, file_path: str) -> Dict[str, Any]:
        """Process Excel files using pandas."""
        logger.info(f"Extracting content from Excel: {os.path.basename(file_path)}")
        result = {'sheets': [], 'text_representation': []}
        all_text_repr: List[str] = []

        try:
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names
            logger.info(f"Excel file has {len(sheet_names)} sheets: {sheet_names}")

            for sheet_name in sheet_names:
                try:
                    # Read sheet without assuming header, keep empty values as None
                    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, keep_default_na=True)

                    # Remove completely empty rows and columns before header detection
                    df.dropna(axis=0, how='all', inplace=True)
                    df.dropna(axis=1, how='all', inplace=True)
                    df.reset_index(drop=True, inplace=True)

                    if df.empty:
                         logger.info(f"Sheet '{sheet_name}' is empty after cleaning, skipping.")
                         continue

                    # Attempt to find a meaningful header row
                    header_row_index = self._find_header_row(df)
                    if header_row_index is not None:
                         logger.debug(f"Detected header row {header_row_index} in sheet '{sheet_name}'.")
                         # Reread with the correct header row
                         df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row_index, keep_default_na=True)
                         # Clean again after re-reading
                         df.dropna(axis=0, how='all', inplace=True)
                         df.dropna(axis=1, how='all', inplace=True)
                    else:
                         logger.warning(f"No clear header found in sheet '{sheet_name}', using default headers.")
                         # Use first row as header if no better one found, or generate generic ones
                         df.columns = [f'Column_{k}' for k in range(len(df.columns))]


                    # Convert numpy NaN/NaT to None for JSON compatibility
                    df = df.replace({np.nan: None, pd.NaT: None})
                    # Convert all data to string for text representation, handling None
                    df_str = df.astype(str).replace({'None': ''})


                    # Store sheet data
                    result['sheets'].append({
                        'name': sheet_name,
                        'data': df.to_dict(orient='records') # Store original types where possible
                    })

                    # Create text representation of the sheet
                    text_rep = f"--- Sheet: {sheet_name} ---\n"
                    # Use tabulate for better formatting if available, otherwise fallback
                    try:
                        from tabulate import tabulate
                        text_rep += tabulate(df_str, headers='keys', tablefmt='grid', showindex=False)
                    except ImportError:
                        text_rep += df_str.to_string(index=False, na_rep='') # Fallback
                    all_text_repr.append(text_rep)

                except Exception as sheet_error:
                     logger.warning(f"Could not process sheet '{sheet_name}' in {os.path.basename(file_path)}: {sheet_error}", exc_info=True)


            # Combine all text representations
            result['combined_text'] = "\n\n".join(all_text_repr)
            logger.info(f"Finished Excel processing. Processed {len(result['sheets'])} sheets. Total text length: {len(result['combined_text'])} chars.")

        except Exception as e:
            logger.error(f"Error processing Excel file '{os.path.basename(file_path)}': {str(e)}")
            raise

        return result

    def _find_header_row(self, df: pd.DataFrame, max_rows_to_check=10) -> Optional[int]:
        """ Tries to find the most likely header row in the first few rows """
        if df.empty:
            return None
        best_score = -1
        best_index = None
        for i in range(min(max_rows_to_check, len(df))):
            row = df.iloc[i]
            score = self._score_header_row(row)
            # logger.debug(f"Sheet row {i} score: {score}, content: {row.tolist()}")
            if score > best_score:
                 best_score = score
                 best_index = i

        # Require a minimum score to consider it a header
        if best_score >= 2: # Adjust threshold as needed
             return best_index
        else:
             return None # Return None if no clear header found

    def _score_header_row(self, row: pd.Series) -> int:
        """ Scores a row based on likelihood of being a header. """
        if row.isnull().all():
            return -1 # Penalize empty rows

        score = 0
        num_numeric = 0
        num_non_numeric = 0
        header_keywords = ['total', 'income', 'expense', 'revenue', 'cost',
                          'date', 'period', 'month', 'year', 'budget', 'actual',
                          'rent', 'fee', 'tax', 'insurance', 'utility', 'maintenance',
                          'admin', 'payroll', 'marketing', 'vacancy', 'concession',
                          'debt', 'recover', 'parking', 'laundry', 'other', 'net',
                          'operating', 'gross', 'potential', 'effective', 'reserve',
                          'improvement', 'allowance', 'item', 'description', 'amount', 'gl', 'account']
        contains_keyword = False
        all_caps_count = 0

        for v in row:
            if pd.isna(v) or str(v).strip() == '':
                continue
            v_str = str(v).strip()
            v_lower = v_str.lower()

            # Check for numeric values
            try:
                cleaned_v = v_lower.replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
                if cleaned_v.strip().startswith('-'):
                    if cleaned_v.strip().endswith('-'): cleaned_v = '-' + cleaned_v.strip().strip('-')
                float(cleaned_v)
                num_numeric += 1
            except ValueError:
                num_non_numeric += 1
                # Check for keywords only in non-numeric cells
                if any(keyword in v_lower for keyword in header_keywords):
                    contains_keyword = True
                # Check for all caps (common in headers)
                if v_str.isupper() and len(v_str) > 1:
                    all_caps_count += 1

        total_valid_cells = num_numeric + num_non_numeric
        if total_valid_cells == 0: return -1

        # Scoring logic (can be tuned)
        if contains_keyword: score += 2
        if num_non_numeric > num_numeric: score += 1
        if num_non_numeric / total_valid_cells > 0.7: score += 1 # Mostly text
        if num_numeric / total_valid_cells < 0.2: score += 1 # Few numbers
        if all_caps_count > total_valid_cells * 0.5: score += 1 # Mostly all caps

        return score


    def _process_csv(self, file_path: str) -> Dict[str, Any]:
        """Process CSV files using pandas."""
        logger.info(f"Extracting content from CSV: {os.path.basename(file_path)}")
        result = {}

        try:
            encoding = self._detect_encoding(file_path)
            logger.info(f"Reading CSV with encoding: {encoding}")

            # Read CSV file, handle potential parsing issues, skip bad lines
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False, on_bad_lines='skip')
            df.dropna(axis=0, how='all', inplace=True)
            df.dropna(axis=1, how='all', inplace=True)

            if df.empty:
                 logger.warning(f"CSV file '{os.path.basename(file_path)}' is empty or contains only invalid lines.")
                 result['data'] = []
                 result['text_representation'] = ""
                 result['combined_text'] = ""
                 return result

            # Convert NaN to None for JSON compatibility
            df = df.replace({np.nan: None, pd.NaT: None})

            # Store data
            result['data'] = df.to_dict(orient='records')

            # Create text representation
            df_str = df.astype(str).replace({'None': ''})
            try:
                from tabulate import tabulate
                result['text_representation'] = tabulate(df_str, headers='keys', tablefmt='grid', showindex=False)
            except ImportError:
                result['text_representation'] = df_str.to_string(index=False, na_rep='') # Fallback

            result['combined_text'] = result['text_representation']
            logger.info(f"Finished CSV processing. Extracted {len(df)} rows. Text length: {len(result['combined_text'])} chars.")


        except Exception as e:
            logger.error(f"Error processing CSV file '{os.path.basename(file_path)}': {str(e)}")
            raise

        return result

    def _process_txt(self, file_path: str) -> Dict[str, Any]:
        """Process TXT files."""
        logger.info(f"Extracting content from TXT: {os.path.basename(file_path)}")
        result = {}

        try:
            encoding = self._detect_encoding(file_path)
            logger.info(f"Reading TXT with encoding: {encoding}")

            # Read text file
            with open(file_path, 'r', encoding=encoding, errors='replace') as f: # Replace errors
                text = f.read()

            # Clean text
            cleaned_text = self._clean_text(text)

            # Store data
            result['text'] = cleaned_text
            result['combined_text'] = cleaned_text
            logger.info(f"Finished TXT processing. Text length: {len(result['combined_text'])} chars.")


        except Exception as e:
            logger.error(f"Error processing TXT file '{os.path.basename(file_path)}': {str(e)}")
            raise

        return result

    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding using chardet."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(20000) # Read more bytes for better detection

            result = chardet.detect(raw_data)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'
            # Handle specific cases like 'ascii' -> 'utf-8' for broader compatibility
            if encoding.lower() == 'ascii':
                encoding = 'utf-8'
            confidence = result['confidence'] if result else 0
            logger.info(f"Detected encoding: {encoding} with confidence {confidence:.2f}")
            # If confidence is very low, default to utf-8 with error handling
            if confidence is None or confidence < 0.7:
                 logger.warning(f"Low confidence ({confidence}) for detected encoding '{encoding}'. Using 'utf-8' as fallback.")
                 # return 'utf-8' # Or stick with low confidence detection? Sticking for now.
            return encoding
        except Exception as e:
             logger.warning(f"Could not detect encoding for {os.path.basename(file_path)}, defaulting to utf-8. Error: {e}")
             return 'utf-8'


    def _clean_text(self, text: str) -> str:
        """Clean and normalize text extracted from documents."""
        if not text:
            return ""

        # Replace multiple whitespace characters (space, tab, newline, etc.) with a single space
        text = re.sub(r'\s+', ' ', text)

        # Normalize line breaks (though less relevant after replacing \s+)
        # text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove leading/trailing whitespace
        text = text.strip()

        # Optional: Remove non-printable characters except common whitespace like \n, \t
        # text = ''.join(char for char in text if char.isprintable() or char in '\n\t')

        return text

    def _is_header_row(self, row: pd.Series) -> bool:
        """Check if a row looks like a header row (calls scoring method)."""
        return self._score_header_row(row) >= 2 # Use score threshold


def preprocess_file(file_path: str, content_type: str = None, filename: str = None) -> Dict[str, Any]:
    """
    Convenience function to preprocess a file.

    Args:
        file_path: Path to the file to preprocess.
        content_type: Content type of the file (optional).
        filename: Original filename (optional).

    Returns:
        Dict containing extracted text/data and metadata.
    """
    preprocessor = FilePreprocessor()
    return preprocessor.preprocess(file_path, content_type, filename)

#################################################
# Document Classifier Module (Unchanged)
#################################################
class DocumentClassifier:
    """
    Class for classifying financial documents and extracting time periods
    using GPT-4, enhanced to work with labeled document types.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the document classifier."""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OpenAI API key not set for DocumentClassifier.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)

        self.document_types = ["Actual Income Statement", "Budget", "Prior Year Actual", "Unknown"]
        self.document_type_mapping = {
            "current_month_actuals": "Actual Income Statement",
            "prior_month_actuals": "Actual Income Statement",
            "current_month_budget": "Budget",
            "prior_year_actuals": "Prior Year Actual"
        }

    def classify(self, text_or_data: Union[str, Dict[str, Any]], known_document_type: Optional[str] = None) -> Dict[str, Any]:
        """Classify document type and extract time period."""
        logger.info("Classifying document and extracting time period...")
        start_time = time.time()

        # 1. Use Known Type if Provided
        if known_document_type:
            mapped_type = self.document_type_mapping.get(known_document_type)
            if mapped_type:
                logger.info(f"Using known document type: '{known_document_type}' mapped to '{mapped_type}'")
                period = self._extract_period_from_content(text_or_data) # Still need period
                return {'document_type': mapped_type, 'period': period, 'method': 'known_type', 'duration_ms': int((time.time() - start_time) * 1000)}
            else:
                logger.warning(f"Unknown mapping for known type '{known_document_type}', proceeding with classification.")

        # 2. Extract Text and Filename
        extracted_text = self._extract_text_from_input(text_or_data)
        filename = text_or_data.get('metadata', {}).get('filename') if isinstance(text_or_data, dict) else None

        # 3. Try Classification from Filename
        if filename:
            doc_type_from_filename = self._determine_type_from_filename(filename)
            if doc_type_from_filename:
                logger.info(f"Determined document type from filename '{filename}': {doc_type_from_filename}")
                period = self._extract_period_from_content(extracted_text) or self._extract_period_from_filename(filename)
                return {'document_type': doc_type_from_filename, 'period': period, 'method': 'filename', 'duration_ms': int((time.time() - start_time) * 1000)}

        # 4. Try Rule-Based Classification
        rule_based_result = self._rule_based_classification(extracted_text)
        if rule_based_result.get('confidence', 0) > 0.75: # Slightly higher threshold
            logger.info(f"Rule-based classification successful: {rule_based_result}")
            return {**rule_based_result, 'method': 'rule_based', 'duration_ms': int((time.time() - start_time) * 1000)}

        # 5. Fallback to GPT Classification
        if not self.client:
             logger.error("OpenAI client not available for GPT classification fallback.")
             # Return rule-based result even if low confidence, or a default unknown
             return {**rule_based_result, 'method': 'rule_based_low_confidence', 'duration_ms': int((time.time() - start_time) * 1000)}

        logger.info("Falling back to GPT for classification.")
        gpt_result = self._gpt_classification(extracted_text)
        logger.info(f"GPT classification result: {gpt_result}")
        return {**gpt_result, 'method': 'gpt', 'duration_ms': int((time.time() - start_time) * 1000)}

    def _extract_text_from_input(self, text_or_data: Union[str, Dict[str, Any]]) -> str:
        """Extract text content from input (string or dictionary)."""
        if isinstance(text_or_data, dict):
            # Prioritize 'combined_text' if available
            if 'combined_text' in text_or_data and isinstance(text_or_data['combined_text'], str):
                return text_or_data['combined_text']
            # Fallback to joining 'text' list content
            elif 'text' in text_or_data and isinstance(text_or_data['text'], list):
                page_contents = [p.get('content', '') if isinstance(p, dict) else str(p) for p in text_or_data['text']]
                return "\n\n".join(filter(None, page_contents))
            # Fallback to other potential keys or string conversion
            elif 'content' in text_or_data and isinstance(text_or_data['content'], str):
                return text_or_data['content']
            else:
                logger.warning("Could not find standard text fields ('combined_text', 'text') in dict, using limited string representation.")
                try:
                    # Attempt limited JSON dump
                    return json.dumps(text_or_data, default=str, ensure_ascii=False, indent=None, separators=(',', ':'))[:5000] # Limit length
                except Exception:
                    return str(text_or_data)[:5000] # Final fallback
        elif isinstance(text_or_data, str):
            return text_or_data
        else:
            logger.warning(f"Input for text extraction is not dict or str: {type(text_or_data)}, converting to string.")
            return str(text_or_data)


    def _extract_period_from_content(self, text_or_data: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extract period information from document content using robust regex."""
        text = self._extract_text_from_input(text_or_data)
        if not text or len(text) < 4: # Need at least 4 chars for a year
            return None

        # Define month names and abbreviations for regex
        months_full = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        months_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        months_pattern = '|'.join(months_full + months_abbr)
        year_pattern = r'(20\d{2})' # Years 2000-2099

        # --- Patterns (Order matters: More specific first) ---

        # 1. "For the Period/Month Ended Month Day, Year" or "Month Day, Year"
        pattern1 = rf'(?:For the (?:Month|Period)\s+(?:Ended|Ending)\s+)?({months_pattern})\s+(\d{{1,2}})[\s,]+{year_pattern}'
        match1 = re.search(pattern1, text, re.IGNORECASE)
        if match1:
            month_str = match1.group(1)
            year_str = match1.group(3)
            month_full = self._standardize_month(month_str, months_full, months_abbr)
            logger.debug(f"Period match pattern 1: {month_full} {year_str}")
            return f"{month_full} {year_str}"

        # 2. "Month Year" (e.g., "January 2025", "Jan 2025", "Jan-2025")
        pattern2 = rf'({months_pattern})[\s.,_-]+{year_pattern}'
        match2 = re.search(pattern2, text, re.IGNORECASE)
        if match2:
            month_str = match2.group(1)
            year_str = match2.group(2)
            month_full = self._standardize_month(month_str, months_full, months_abbr)
            logger.debug(f"Period match pattern 2: {month_full} {year_str}")
            return f"{month_full} {year_str}"

        # 3. Quarter Year (e.g., "Q1 2025", "First Quarter 2025", "Quarter Ended Mar 31, 2025")
        quarter_words = ['First', 'Second', 'Third', 'Fourth']
        pattern3 = rf'(?:(Q[1-4])|({"|".join(quarter_words)})\s+Quarter)[\s.,_-]*{year_pattern}|(?:Quarter\s+(?:Ended|Ending)\s+(?:{months_pattern})\s+\d{{1,2}}[\s.,_-]*)({year_pattern})'
        match3 = re.search(pattern3, text, re.IGNORECASE)
        if match3:
            year_str = match3.group(3) or match3.group(4) # Year from QX YYYY or Quarter Ended YYYY
            if match3.group(1): # Q1-Q4 format
                period_str = f"{match3.group(1)} {year_str}"
            elif match3.group(2): # First-Fourth Quarter format
                 period_str = f"{match3.group(2)} Quarter {year_str}"
            else: # Quarter Ended ... YYYY format
                 period_str = f"Quarter in {year_str}" # Less specific
            logger.debug(f"Period match pattern 3: {period_str}")
            return period_str

        # 4. Year Only (e.g., "For the Year Ended...", "FY 2025", "Calendar Year 2025")
        pattern4 = rf'(?:For the Year\s+(?:Ended|Ending)|FY|Fiscal Year|Calendar Year)[\s.,_-]*({year_pattern})'
        match4 = re.search(pattern4, text, re.IGNORECASE)
        if match4:
            period_str = f"Year {match4.group(1)}"
            logger.debug(f"Period match pattern 4: {period_str}")
            return period_str

        # 5. Fallback: Standalone Year in context
        for match5 in re.finditer(rf'\b({year_pattern})\b', text):
            year_str = match5.group(1)
            context_window = text[max(0, match5.start()-40):min(len(text), match5.end()+40)]
            # Check for keywords indicating it's likely *the* period year
            if any(kw in context_window.lower() for kw in ['period', 'month', 'year', 'date', 'quarter', 'budget', 'actual', 'statement', 'report', 'ended', 'ending']):
                 # Try to find month nearby for more specificity
                 month_nearby_match = re.search(rf'({months_pattern})', context_window, re.IGNORECASE)
                 if month_nearby_match:
                      month_str = month_nearby_match.group(1)
                      month_full = self._standardize_month(month_str, months_full, months_abbr)
                      period_str = f"{month_full} {year_str}"
                      logger.debug(f"Period match pattern 5 (Year+Month Context): {period_str}")
                      return period_str
                 else:
                      period_str = f"Year {year_str}"
                      logger.debug(f"Period match pattern 5 (Year Context): {period_str}")
                      return period_str # Return year if month not nearby

        logger.warning("Could not extract specific period from content using regex.")
        return None # Return None if no specific period found

    def _standardize_month(self, month_str: str, months_full: List[str], months_abbr: List[str]) -> str:
        """Converts abbreviated month to full month name."""
        month_lower = month_str.lower()
        for i, abbr in enumerate(months_abbr):
            if month_lower.startswith(abbr.lower()):
                return months_full[i]
        return month_str # Return original if no match (shouldn't happen with pattern)

    def _determine_type_from_filename(self, filename: Optional[str]) -> Optional[str]:
        """Determine document type from filename keywords."""
        if not filename: return None
        filename_lower = filename.lower()
        is_budget = 'budget' in filename_lower or 'bdgt' in filename_lower
        is_actual = 'actual' in filename_lower or 'is' in filename_lower or 'p&l' in filename_lower or 'income statement' in filename_lower or 'profit loss' in filename_lower
        is_prior_year = 'prior year' in filename_lower or 'py' in filename_lower or 'previous year' in filename_lower or 'last year' in filename_lower

        if is_budget: return "Budget"
        if is_actual:
            if is_prior_year: return "Prior Year Actual"
            # Could add prior month check here if needed:
            # elif 'prior month' in filename_lower or 'pm' in filename_lower: return "Actual Income Statement" # Or a specific type
            else: return "Actual Income Statement"
        return None

    def _extract_period_from_filename(self, filename: str) -> Optional[str]:
        """Extract period information from filename (e.g., "IS_Jan_2025.pdf")."""
        if not filename: return None
        name_part = os.path.splitext(filename)[0]
        name_part = re.sub(r'[_\-\.]+', ' ', name_part) # Normalize separators

        months_full = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        months_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        months_pattern = '|'.join(months_full + months_abbr)
        year_pattern = r'(20\d{2})'

        # Try Month Year
        month_year_match = re.search(rf'({months_pattern})\s+{year_pattern}', name_part, re.IGNORECASE)
        if month_year_match:
            month_full = self._standardize_month(month_year_match.group(1), months_full, months_abbr)
            return f"{month_full} {month_year_match.group(2)}"

        # Try Year Month
        year_month_match = re.search(rf'{year_pattern}\s+({months_pattern})', name_part, re.IGNORECASE)
        if year_month_match:
            month_full = self._standardize_month(year_month_match.group(2), months_full, months_abbr)
            return f"{month_full} {year_month_match.group(1)}"

        # Try Quarter Year
        quarter_match = re.search(rf'(Q[1-4])\s*{year_pattern}', name_part, re.IGNORECASE)
        if quarter_match: return f"{quarter_match.group(1)} {quarter_match.group(2)}"
        quarter_match_rev = re.search(rf'{year_pattern}\s*(Q[1-4])', name_part, re.IGNORECASE)
        if quarter_match_rev: return f"{quarter_match_rev.group(2)} {quarter_match_rev.group(1)}"

        # Try Year only
        year_match = re.search(year_pattern, name_part)
        if year_match and any(kw in name_part.lower() for kw in ['budget', 'actual', 'year', 'report']):
            return f"Year {year_match.group(1)}"

        return None

    def _rule_based_classification(self, text: str) -> Dict[str, Any]:
        """Attempt rule-based classification."""
        result = {'document_type': 'Unknown', 'period': None, 'confidence': 0.0}
        if not text: return result

        text_lower = text.lower()
        doc_type = 'Unknown'
        type_confidence = 0.0

        # Score based on keywords
        budget_score = text_lower.count('budget') + text_lower.count('variance')
        actual_score = text_lower.count('actual') + text_lower.count('income statement') + text_lower.count('profit and loss')
        prior_year_score = text_lower.count('prior year') + text_lower.count('previous year')

        if budget_score > actual_score and budget_score > 0:
            doc_type = 'Budget'
            type_confidence = min(0.8 + budget_score * 0.1, 1.0)
        elif actual_score > 0:
            if prior_year_score > 0:
                doc_type = 'Prior Year Actual'
                type_confidence = min(0.8 + prior_year_score * 0.1, 1.0)
            else:
                doc_type = 'Actual Income Statement'
                type_confidence = min(0.7 + actual_score * 0.05, 0.9) # Slightly lower base

        result['document_type'] = doc_type
        result['confidence'] = type_confidence

        period = self._extract_period_from_content(text)
        if period:
            result['period'] = period
            result['confidence'] = min(result['confidence'] + 0.15, 1.0) # Boost confidence if period found

        return result

    def _gpt_classification(self, text: str) -> Dict[str, Any]:
        """Classify document using GPT-4."""
        default_result = {'document_type': 'Unknown', 'period': None}
        if not self.client:
            logger.error("OpenAI client not initialized. Cannot perform GPT classification.")
            return default_result

        text_sample = text[:3000] if text else "" # Slightly larger sample
        if not text_sample:
             logger.warning("Empty text provided for GPT classification.")
             return default_result

        prompt = f"""Analyze the beginning of this financial document to determine its type and the period it covers.

Document Types:
- Actual Income Statement (Current or Prior Month)
- Budget
- Prior Year Actual
- Unknown

Period Format: Extract the most specific period available (e.g., "Month Year", "Quarter Year", "Year").

Document Sample:
---
{text_sample}
---

Respond ONLY with a JSON object containing 'document_type' and 'period' fields. Example: {{"document_type": "Actual Income Statement", "period": "March 2025"}}
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4", # Or "gpt-3.5-turbo"
                messages=[
                    {"role": "system", "content": "You are an expert financial document classifier. Respond only in the requested JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1, max_tokens=100
            )
            response_text = response.choices[0].message.content.strip()
            logger.info(f"GPT classification raw response: {response_text}")

            try:
                cleaned_response_text = response_text.strip('`').strip()
                if cleaned_response_text.startswith('json'): cleaned_response_text = cleaned_response_text[4:].strip()
                result = json.loads(cleaned_response_text)

                if 'document_type' in result and 'period' in result:
                     if result['document_type'] not in self.document_types:
                          logger.warning(f"GPT returned unknown document type: {result['document_type']}")
                          result['document_type'] = 'Unknown'
                     logger.info(f"Successfully parsed GPT classification response: {result}")
                     return result
                else:
                     logger.error(f"GPT classification response missing required fields: {cleaned_response_text}")
                     raise json.JSONDecodeError("Missing fields", cleaned_response_text, 0)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response from GPT classification: {str(e)}")
                # Fallback regex (less reliable)
                doc_type_match = re.search(r'"document_type":\s*"([^"]+)"', response_text)
                period_match = re.search(r'"period":\s*"([^"]*)"', response_text)
                doc_type_res = doc_type_match.group(1) if doc_type_match else 'Unknown'
                period_res = period_match.group(1) if period_match else None
                if doc_type_res not in self.document_types: doc_type_res = 'Unknown'
                logger.warning(f"Falling back to regex extraction for classification: Type='{doc_type_res}', Period='{period_res}'")
                return {'document_type': doc_type_res, 'period': period_res}
        except (RateLimitError, APIError, APITimeoutError) as api_err:
             logger.error(f"OpenAI API error during classification: {api_err}")
             return default_result # Return default on API errors
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI API for classification: {str(e)}", exc_info=True)
            return default_result

# Create an instance of the classifier for direct import
classifier = DocumentClassifier()

# Convenience functions
def classify_document(text_or_data: Union[str, Dict[str, Any]], known_document_type: Optional[str] = None) -> Tuple[str, Optional[str]]:
    result = classifier.classify(text_or_data, known_document_type)
    return result['document_type'], result['period']

def map_noi_tool_to_extraction_type(noi_tool_type: str) -> str:
    return classifier.document_type_mapping.get(noi_tool_type, "Unknown")


#################################################
# GPT Data Extractor Module (MERGED: Detailed NOI + Application Fees)
#################################################
class GPTDataExtractor:
    """
    Class for extracting detailed financial data (incl. Application Fees) using GPT-4.
    """

    def __init__(self, api_key: Optional[str] = None, sample_limit: int = 4000):
        """Initialize the data extractor."""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.sample_limit = sample_limit

        if not self.api_key:
            logger.warning("OpenAI API key not set for GPTDataExtractor.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)

    def extract(self, text_or_data: Union[str, Dict[str, Any]], document_type: str, period: Optional[str] = None) -> Dict[str, Any]:
        """Extracts detailed financial data."""
        if not self.client:
             logger.error("OpenAI client not initialized. Cannot extract data.")
             return self._empty_result(document_type, period)

        logger.info(f"Extracting detailed financial data from {document_type} for period {period or 'Unknown'}")
        text = classifier._extract_text_from_input(text_or_data) # Use classifier's text extraction
        if not text:
             logger.warning("No text content found for extraction.")
             return self._empty_result(document_type, period)

        prompt = self._create_extraction_prompt(text, document_type, period)
        extraction_result = self._extract_with_gpt(prompt)

        # Ensure basic structure and types exist even if GPT fails partially
        if not extraction_result or not isinstance(extraction_result, dict):
             logger.warning("GPT extraction returned invalid or empty result. Using default structure.")
             extraction_result = self._empty_result(document_type, period)
        else:
             # Ensure essential keys are present before validation
             extraction_result['document_type'] = extraction_result.get('document_type', document_type)
             extraction_result['period'] = extraction_result.get('period', period or 'Unknown')
             # Ensure all expected numeric keys exist before cleaning
             default_empty = self._empty_result()
             for key in default_empty.keys():
                 if key not in ['document_type', 'period'] and key not in extraction_result:
                     extraction_result[key] = default_empty[key]


        # Perform validation and cleaning *after* getting the result
        self._validate_and_clean_extraction_result(extraction_result)
        return extraction_result


    def _create_extraction_prompt(self, text: str, document_type: str, period: Optional[str] = None) -> str:
        """Creates the detailed prompt for GPT-4 extraction, including Application Fees."""
        text_sample = text[:self.sample_limit]
        period_context = f" for the period '{period}'" if period else ""

        # --- THIS IS THE MERGED PROMPT ---
        prompt = f"""You are an expert real estate financial analyst. Extract detailed financial data from the following '{document_type}' document{period_context}.

        Document Text Sample:
        ---
        {text_sample}
        ---

        Extract the following financial metrics precisely. If a specific line item isn't present, use 0. If a total is provided, use it but verify against components if possible. If a total is not provided, calculate it from the components you find.

        1.  **POTENTIAL GROSS INCOME (PGI):**
            * `gross_potential_rent`: Scheduled base rent amount.
            * `loss_to_lease`: Difference between market rent and actual contract rent.

        2.  **VACANCY & CREDIT LOSS:**
            * `physical_vacancy_loss`: Income lost due to unoccupied units.
            * `concessions_free_rent`: Value of rent discounts or free periods given.
            * `bad_debt`: Uncollectible rent.
            * `total_vacancy_credit_loss`: Sum of the above three items (calculate if not explicitly stated).

        3.  **OTHER INCOME:**
            * `recoveries`: CAM, Tax, Insurance reimbursements from tenants.
            * `parking_income`: Income from parking fees.
            * `laundry_income`: Income from laundry facilities.
            * `application_fees`: Fees collected from rental applications. <--- ADDED
            * `other_misc_income`: Other income (storage, signage, late fees, etc.). Sum if multiple lines exist.
            * `total_other_income`: Sum of ALL above other income items (calculate if not explicitly stated). <--- UPDATED CALC

        4.  **EFFECTIVE GROSS INCOME (EGI):**
            * `effective_gross_income`: Calculate as (Gross Potential Rent - Total Vacancy & Credit Loss + Total Other Income). Verify if stated explicitly.

        5.  **OPERATING EXPENSES (OpEx):**
            * `property_taxes`: Real estate taxes.
            * `insurance`: Property insurance costs.
            * `property_management_fees`: Fees paid for management.
            * `repairs_maintenance`: General repairs and maintenance costs.
            * `utilities`: Sum of all utility costs (electric, gas, water/sewer, trash). If breakdown is obvious, capture the total.
            * `payroll`: Salaries, wages, benefits for on-site staff.
            * `admin_office_costs`: General administrative and office expenses.
            * `marketing_advertising`: Costs for advertising and marketing.
            * `other_opex`: Other operating expenses not listed above. Sum if multiple lines exist.
            * `total_operating_expenses`: Sum of all the above expense items (calculate if not explicitly stated).

        6.  **NET OPERATING INCOME (NOI):**
            * `net_operating_income`: Calculate as (Effective Gross Income - Total Operating Expenses). Verify if stated explicitly.

        7.  **RESERVES (Below NOI - Optional):**
            * `replacement_reserves`: Funds set aside for future capital replacements (e.g., roof, HVAC).
            * `tenant_improvements`: Costs for tenant-specific build-outs or improvements.

        Respond ONLY with a JSON object containing these fields. Use 0 where values are not found or not applicable. Ensure all values are numbers.

        JSON Structure Example:
        {{
          "document_type": "{document_type}",
          "period": "{period or 'Unknown'}",
          "gross_potential_rent": 0,
          "loss_to_lease": 0,
          "physical_vacancy_loss": 0,
          "concessions_free_rent": 0,
          "bad_debt": 0,
          "total_vacancy_credit_loss": 0,
          "recoveries": 0,
          "parking_income": 0,
          "laundry_income": 0,
          "application_fees": 0,          # <--- ADDED
          "other_misc_income": 0,
          "total_other_income": 0,
          "effective_gross_income": 0,
          "property_taxes": 0,
          "insurance": 0,
          "property_management_fees": 0,
          "repairs_maintenance": 0,
          "utilities": 0,
          "payroll": 0,
          "admin_office_costs": 0,
          "marketing_advertising": 0,
          "other_opex": 0,
          "total_operating_expenses": 0,
          "net_operating_income": 0,
          "replacement_reserves": 0,
          "tenant_improvements": 0
        }}
        """
        # --- END OF MERGED PROMPT ---
        return prompt

    def _extract_with_gpt(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Sends prompt to GPT and parses the JSON response."""
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model="gpt-4", # Or "gpt-4-turbo"
                messages=[
                    {"role": "system", "content": "You are an expert financial data extraction AI. Respond only in the requested JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1, # Low temperature for consistency
                max_tokens=1500 # Increased tokens for potentially larger JSON
            )
            duration = time.time() - start_time
            logger.info(f"GPT extraction response time: {duration:.2f}s")

            response_text = response.choices[0].message.content.strip()
            logger.debug(f"GPT raw extraction response: {response_text}")

            # Attempt to parse JSON, cleaning potential markdown/prefix issues
            try:
                # Remove potential markdown code block fences and json prefix
                cleaned_response_text = response_text.strip()
                if cleaned_response_text.startswith("```json"):
                    cleaned_response_text = cleaned_response_text[7:]
                if cleaned_response_text.endswith("```"):
                    cleaned_response_text = cleaned_response_text[:-3]
                cleaned_response_text = cleaned_response_text.strip()

                result = json.loads(cleaned_response_text)
                logger.info("Successfully parsed JSON response from GPT extraction.")
                # Basic check if it's a dictionary
                if isinstance(result, dict):
                    return result
                else:
                    logger.error(f"Parsed result is not a dictionary: {type(result)}")
                    return None
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response from GPT extraction: {str(e)}")
                # Fallback: Try to find JSON within the text if parsing fails
                json_match = re.search(r'\{.*\}', cleaned_response_text, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(0))
                        if isinstance(result, dict):
                             logger.warning("Successfully parsed JSON using regex fallback.")
                             return result
                        else:
                             logger.error(f"Regex fallback result is not a dictionary: {type(result)}")
                             return None
                    except json.JSONDecodeError:
                        logger.error("Error parsing JSON even with regex fallback.")
                else:
                     logger.error("No JSON object found in the response.")
                return None # Return None if parsing fails

        except (RateLimitError, APIError, APITimeoutError) as api_err:
             logger.error(f"OpenAI API error during extraction: {api_err}")
             return None # Return None on API errors
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI API for extraction: {str(e)}", exc_info=True)
            return None # Return None on other errors

    def _clean_numeric_value(self, value: Any) -> float:
        """Attempts to clean and convert a value to a float."""
        if isinstance(value, (int, float)):
            # Handle potential numpy types just in case
            if isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                                  np.uint8, np.uint16, np.uint32, np.uint64)):
                 return float(value)
            if isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                 # Check for NaN before converting
                 return float(value) if not np.isnan(value) else 0.0
            # Standard int/float
            return float(value)

        if isinstance(value, str):
            # Remove common currency symbols, commas, spaces
            cleaned_value = value.strip().replace('$', '').replace(',', '').replace(' ', '')
            # Handle parentheses for negatives: (100.00) -> -100.00
            if cleaned_value.startswith('(') and cleaned_value.endswith(')'):
                cleaned_value = '-' + cleaned_value[1:-1]
            # Handle trailing minus sign: 100.00- -> -100.00
            if cleaned_value.endswith('-'):
                cleaned_value = '-' + cleaned_value[:-1]
            # Handle empty string after cleaning
            if not cleaned_value:
                return 0.0
            try:
                return float(cleaned_value)
            except (ValueError, TypeError):
                # logger.debug(f"Could not convert string to float: '{value}'")
                return 0.0 # Default to 0 if conversion fails
        # logger.debug(f"Could not convert non-string/numeric type to float: {type(value)}")
        return 0.0 # Default non-numeric/non-string types to 0

    def _validate_and_clean_extraction_result(self, result: Dict[str, Any]) -> None:
        """Validates extracted data, cleans numeric fields, and recalculates totals."""
        if not result or not isinstance(result, dict):
            logger.warning("Skipping validation for invalid or empty extraction result.")
            return

        logger.debug("Starting validation and cleaning of extraction result.")
        # Define all expected numeric fields from the MERGED prompt
        numeric_fields = [
            'gross_potential_rent', 'loss_to_lease', 'physical_vacancy_loss',
            'concessions_free_rent', 'bad_debt', 'total_vacancy_credit_loss',
            'recoveries', 'parking_income', 'laundry_income', 'application_fees', # Added application_fees
            'other_misc_income', 'total_other_income', 'effective_gross_income',
            'property_taxes', 'insurance', 'property_management_fees',
            'repairs_maintenance', 'utilities', 'payroll', 'admin_office_costs',
            'marketing_advertising', 'other_opex', 'total_operating_expenses',
            'net_operating_income', 'replacement_reserves', 'tenant_improvements'
        ]

        # Clean all expected numeric fields first
        for field in numeric_fields:
            # Use .get() for safety, default to 0 before cleaning if key missing
            result[field] = self._clean_numeric_value(result.get(field, 0.0))

        # --- Recalculate and Validate Totals ---
        tolerance = 1.0 # Allow $1 difference for rounding

        # 1. Vacancy & Credit Loss
        calc_vacancy_loss = sum([result.get('physical_vacancy_loss', 0.0),
                                 result.get('concessions_free_rent', 0.0),
                                 result.get('bad_debt', 0.0)])
        reported_vacancy_loss = result.get('total_vacancy_credit_loss', 0.0)
        if abs(calc_vacancy_loss - reported_vacancy_loss) > tolerance:
            logger.warning(f"Vacancy Loss mismatch: calculated={calc_vacancy_loss:.2f}, reported={reported_vacancy_loss:.2f}. Using calculated value.")
            result['total_vacancy_credit_loss'] = calc_vacancy_loss

        # 2. Other Income (Now includes application_fees)
        calc_other_income = sum([result.get('recoveries', 0.0),
                                 result.get('parking_income', 0.0),
                                 result.get('laundry_income', 0.0),
                                 result.get('application_fees', 0.0), # Include application_fees
                                 result.get('other_misc_income', 0.0)])
        reported_other_income = result.get('total_other_income', 0.0)
        if abs(calc_other_income - reported_other_income) > tolerance:
            logger.warning(f"Other Income mismatch: calculated={calc_other_income:.2f}, reported={reported_other_income:.2f}. Using calculated value.")
            result['total_other_income'] = calc_other_income

        # 3. Effective Gross Income (EGI)
        calc_egi = (result.get('gross_potential_rent', 0.0) -
                    result.get('total_vacancy_credit_loss', 0.0) +
                    result.get('total_other_income', 0.0))
        reported_egi = result.get('effective_gross_income', 0.0)
        if abs(calc_egi - reported_egi) > tolerance:
             logger.warning(f"EGI mismatch: calculated={calc_egi:.2f} (GPR-Vac+Other), reported={reported_egi:.2f}. Using calculated value.")
             result['effective_gross_income'] = calc_egi

        # 4. Total Operating Expenses
        opex_components = [
            'property_taxes', 'insurance', 'property_management_fees',
            'repairs_maintenance', 'utilities', 'payroll',
            'admin_office_costs', 'marketing_advertising', 'other_opex'
        ]
        calc_total_opex = sum(result.get(field, 0.0) for field in opex_components)
        reported_total_opex = result.get('total_operating_expenses', 0.0)
        if abs(calc_total_opex - reported_total_opex) > tolerance:
            logger.warning(f"Total OpEx mismatch: calculated={calc_total_opex:.2f}, reported={reported_total_opex:.2f}. Using calculated value.")
            result['total_operating_expenses'] = calc_total_opex

        # 5. Net Operating Income (NOI)
        calc_noi = result.get('effective_gross_income', 0.0) - result.get('total_operating_expenses', 0.0)
        reported_noi = result.get('net_operating_income', 0.0)
        if abs(calc_noi - reported_noi) > tolerance:
            logger.warning(f"NOI mismatch: calculated={calc_noi:.2f} (EGI-OpEx), reported={reported_noi:.2f}. Using calculated value.")
            result['net_operating_income'] = calc_noi

        logger.debug("Finished validation and cleaning.")


    def _empty_result(self, document_type: str = "Unknown", period: Optional[str] = None) -> Dict[str, Any]:
        """Returns a default dictionary structure with zeros for all financial fields."""
        return {
            "document_type": document_type,
            "period": period or 'Unknown',
            "gross_potential_rent": 0.0,
            "loss_to_lease": 0.0,
            "physical_vacancy_loss": 0.0,
            "concessions_free_rent": 0.0,
            "bad_debt": 0.0,
            "total_vacancy_credit_loss": 0.0,
            "recoveries": 0.0,
            "parking_income": 0.0,
            "laundry_income": 0.0,
            "application_fees": 0.0, # Added application_fees
            "other_misc_income": 0.0,
            "total_other_income": 0.0,
            "effective_gross_income": 0.0,
            "property_taxes": 0.0,
            "insurance": 0.0,
            "property_management_fees": 0.0,
            "repairs_maintenance": 0.0,
            "utilities": 0.0,
            "payroll": 0.0,
            "admin_office_costs": 0.0,
            "marketing_advertising": 0.0,
            "other_opex": 0.0,
            "total_operating_expenses": 0.0,
            "net_operating_income": 0.0,
            "replacement_reserves": 0.0,
            "tenant_improvements": 0.0
        }

# Create an instance for potential direct use (though API is primary)
extractor = GPTDataExtractor()

def extract_financial_data(text_or_data: Union[str, Dict[str, Any]], document_type: str, period: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to call the extractor."""
    return extractor.extract(text_or_data, document_type, period)

#################################################
# Validation Formatter Module (MERGED: Detailed NOI + Application Fees)
#################################################
class ValidationFormatter:
    """
    Class for validating and formatting extracted detailed financial data (incl. Application Fees).
    """

    def __init__(self):
        """Initialize the validation and formatting module"""
        # Define expected fields and their types (reflecting the merged structure)
        self.expected_fields = {
            'document_type': str,
            'period': str,
            'gross_potential_rent': (float, int),
            'loss_to_lease': (float, int),
            'physical_vacancy_loss': (float, int),
            'concessions_free_rent': (float, int),
            'bad_debt': (float, int),
            'total_vacancy_credit_loss': (float, int),
            'recoveries': (float, int),
            'parking_income': (float, int),
            'laundry_income': (float, int),
            'application_fees': (float, int), # Added application_fees
            'other_misc_income': (float, int),
            'total_other_income': (float, int),
            'effective_gross_income': (float, int),
            'property_taxes': (float, int),
            'insurance': (float, int),
            'property_management_fees': (float, int),
            'repairs_maintenance': (float, int),
            'utilities': (float, int), # Allow dict later if needed
            'payroll': (float, int),
            'admin_office_costs': (float, int),
            'marketing_advertising': (float, int),
            'other_opex': (float, int),
            'total_operating_expenses': (float, int),
            'net_operating_income': (float, int),
            'replacement_reserves': (float, int),
            'tenant_improvements': (float, int)
        }

        # Define validation rules (updated for merged structure)
        self.validation_rules = [
            self._validate_document_type,
            self._validate_period,
            self._validate_numeric_fields, # Checks basic types
            self._validate_vacancy_sum,
            self._validate_other_income_sum, # Updated to include application_fees
            self._validate_egi,
            self._validate_total_opex_sum,
            self._validate_noi_from_egi
        ]

    def validate_and_format(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate and format the extracted detailed financial data.

        Args:
            data: Extracted financial data (should match structure from GPT extractor).

        Returns:
            Tuple containing:
                - Formatted data in the new required JSON structure.
                - List of validation warnings.
        """
        logger.info("Validating and formatting detailed financial data (incl. App Fees)")

        warnings = []
        # Start with the data from extractor (already cleaned and validated there)
        validated_data = data.copy()

        # --- Run final validation rules on the already cleaned data ---
        # These act as sanity checks on the structure provided by the extractor
        for rule in self.validation_rules:
            try:
                 rule_warnings = rule(validated_data)
                 warnings.extend(rule_warnings)
            except Exception as e:
                 logger.error(f"Error during validation rule {rule.__name__}: {e}", exc_info=True)
                 warnings.append(f"Internal error during validation: {rule.__name__}")


        # Format the data according to the new required structure
        final_output = self._format_output(validated_data)

        # Add warnings to the output if any occurred during this final validation pass
        if warnings:
            final_output['validation_warnings'] = warnings
            logger.warning(f"Final validation warnings generated: {warnings}")


        return final_output, warnings # Return formatted data and warnings separately

    def _clean_value_for_validation(self, data: Dict[str, Any], field: str) -> float:
         """ Safely gets a numeric value, assuming it was already cleaned by extractor """
         # Use .get with default 0.0, as extractor should have populated and cleaned
         value = data.get(field, 0.0)
         # Final check for None or non-numeric types just in case
         if not isinstance(value, (int, float)):
              logger.warning(f"Validation found non-numeric value for '{field}': {value}. Using 0.0.")
              return 0.0
         return float(value)


    def _validate_document_type(self, data: Dict[str, Any]) -> List[str]:
        """Validate document_type field"""
        warnings = []
        valid_types = classifier.document_types # Use types from classifier
        doc_type = data.get('document_type')
        if not isinstance(doc_type, str) or doc_type not in valid_types:
            warnings.append(f"Invalid document_type: {doc_type}. Expected one of {valid_types}")
            data['document_type'] = 'Unknown' # Correct invalid type
        return warnings

    def _validate_period(self, data: Dict[str, Any]) -> List[str]:
        """Validate period field"""
        warnings = []
        period = data.get('period')
        if not period or not isinstance(period, str):
            warnings.append(f"Missing or invalid period: {period}")
            data['period'] = 'Unknown' # Correct invalid period
        return warnings

    def _validate_numeric_fields(self, data: Dict[str, Any]) -> List[str]:
        """Validate numeric fields are indeed numeric after cleaning by extractor"""
        warnings = []
        for field, field_type in self.expected_fields.items():
            if field_type in [(float, int), float, int]:
                value = data.get(field)
                # Allow None only for optional reserves
                if field in ['replacement_reserves', 'tenant_improvements'] and value is None:
                     continue
                if not isinstance(value, (float, int)):
                     warnings.append(f"Field {field} ('{value}') is not numeric type after cleaning.")
                     data[field] = 0.0 # Default non-numeric to 0
        return warnings

    # --- Updated/New Validation Rules ---

    def _validate_vacancy_sum(self, data: Dict[str, Any]) -> List[str]:
        """Validate total_vacancy_credit_loss against its components"""
        warnings = []
        tolerance = 1.0
        components = [
            self._clean_value_for_validation(data, 'physical_vacancy_loss'),
            self._clean_value_for_validation(data, 'concessions_free_rent'),
            self._clean_value_for_validation(data, 'bad_debt')
        ]
        calculated_sum = sum(components)
        reported_total = self._clean_value_for_validation(data, 'total_vacancy_credit_loss')
        if abs(calculated_sum - reported_total) > tolerance:
            warnings.append(f"Validation Check: Total Vacancy/Credit Loss ({reported_total:.2f}) differs from sum of components ({calculated_sum:.2f}). Extractor's value used.")
        return warnings

    def _validate_other_income_sum(self, data: Dict[str, Any]) -> List[str]:
        """Validate total_other_income against its components (incl. application_fees)"""
        warnings = []
        tolerance = 1.0
        components = [
            self._clean_value_for_validation(data, 'recoveries'),
            self._clean_value_for_validation(data, 'parking_income'),
            self._clean_value_for_validation(data, 'laundry_income'),
            self._clean_value_for_validation(data, 'application_fees'), # Include application_fees
            self._clean_value_for_validation(data, 'other_misc_income')
        ]
        calculated_sum = sum(components)
        reported_total = self._clean_value_for_validation(data, 'total_other_income')
        if abs(calculated_sum - reported_total) > tolerance:
            warnings.append(f"Validation Check: Total Other Income ({reported_total:.2f}) differs from sum of components ({calculated_sum:.2f}). Extractor's value used.")
        return warnings

    def _validate_egi(self, data: Dict[str, Any]) -> List[str]:
        """Validate effective_gross_income calculation"""
        warnings = []
        tolerance = 1.0
        gpr = self._clean_value_for_validation(data, 'gross_potential_rent')
        vacancy = self._clean_value_for_validation(data, 'total_vacancy_credit_loss')
        other_income = self._clean_value_for_validation(data, 'total_other_income')
        calculated_egi = gpr - vacancy + other_income
        reported_egi = self._clean_value_for_validation(data, 'effective_gross_income')
        if abs(calculated_egi - reported_egi) > tolerance:
             warnings.append(f"Validation Check: EGI ({reported_egi:.2f}) differs from calculated (GPR-Vacancy+OtherIncome = {calculated_egi:.2f}). Extractor's value used.")
        return warnings

    def _validate_total_opex_sum(self, data: Dict[str, Any]) -> List[str]:
        """Validate total_operating_expenses against its components"""
        warnings = []
        tolerance = 1.0
        opex_components = [
            'property_taxes', 'insurance', 'property_management_fees',
            'repairs_maintenance', 'utilities', 'payroll',
            'admin_office_costs', 'marketing_advertising', 'other_opex'
        ]
        calculated_sum = sum(self._clean_value_for_validation(data, field) for field in opex_components)
        reported_total = self._clean_value_for_validation(data, 'total_operating_expenses')
        if abs(calculated_sum - reported_total) > tolerance:
            warnings.append(f"Validation Check: Total OpEx ({reported_total:.2f}) differs from sum of components ({calculated_sum:.2f}). Extractor's value used.")
        return warnings

    def _validate_noi_from_egi(self, data: Dict[str, Any]) -> List[str]:
        """Validate net_operating_income based on EGI and Total OpEx"""
        warnings = []
        tolerance = 1.0
        egi = self._clean_value_for_validation(data, 'effective_gross_income')
        total_opex = self._clean_value_for_validation(data, 'total_operating_expenses')
        calculated_noi = egi - total_opex
        reported_noi = self._clean_value_for_validation(data, 'net_operating_income')
        if abs(calculated_noi - reported_noi) > tolerance:
            warnings.append(f"Validation Check: NOI ({reported_noi:.2f}) differs from calculated (EGI-TotalOpEx = {calculated_noi:.2f}). Extractor's value used.")
        return warnings


    def _format_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the data according to the new required JSON structure with nested objects,
        including application_fees. Uses cleaned numeric values from the input data.

        Args:
            data: Validated financial data dictionary (already cleaned by extractor).

        Returns:
            Formatted data in the required JSON structure.
        """
        # Data should already be cleaned, just structure it
        output = {
            "document_type": data.get('document_type', 'Unknown'),
            "period": data.get('period', 'Unknown'),
            "financials": {
                "income_summary": {
                    "gross_potential_rent": data.get('gross_potential_rent'),
                    "loss_to_lease": data.get('loss_to_lease'),
                    "vacancy_credit_loss_details": {
                        "physical_vacancy_loss": data.get('physical_vacancy_loss'),
                        "concessions_free_rent": data.get('concessions_free_rent'),
                        "bad_debt": data.get('bad_debt'),
                    },
                    "total_vacancy_credit_loss": data.get('total_vacancy_credit_loss'),
                    "other_income_details": {
                        "recoveries": data.get('recoveries'),
                        "parking_income": data.get('parking_income'),
                        "laundry_income": data.get('laundry_income'),
                        "application_fees": data.get('application_fees'), # Added application_fees
                        "other_misc_income": data.get('other_misc_income'),
                    },
                    "total_other_income": data.get('total_other_income'),
                    "effective_gross_income": data.get('effective_gross_income')
                },
                "operating_expenses": {
                    "expense_details": {
                        "property_taxes": data.get('property_taxes'),
                        "insurance": data.get('insurance'),
                        "property_management_fees": data.get('property_management_fees'),
                        "repairs_maintenance": data.get('repairs_maintenance'),
                        "utilities": data.get('utilities'),
                        "payroll": data.get('payroll'),
                        "admin_office_costs": data.get('admin_office_costs'),
                        "marketing_advertising": data.get('marketing_advertising'),
                        "other_opex": data.get('other_opex'),
                    },
                    "total_operating_expenses": data.get('total_operating_expenses'),
                },
                "net_operating_income": data.get('net_operating_income'),
                "reserves": { # Keep reserves section, values might be 0
                    "replacement_reserves": data.get('replacement_reserves'),
                    "tenant_improvements": data.get('tenant_improvements')
                }
            }
        }
        # No need to remove reserves section, 0 is valid data.
        return output

def validate_and_format_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to validate and format extracted detailed financial data.

    Args:
        data: Extracted financial data from the GPT extractor.

    Returns:
        Formatted data in the required JSON structure, potentially including validation warnings.
    """
    validator = ValidationFormatter()
    formatted_data, _ = validator.validate_and_format(data)
    # Warnings are logged within the method and added to formatted_data if present
    return formatted_data

#################################################
# API Server Module (Updated Pydantic Models)
#################################################
# Get API key from environment variable
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    logger.warning("API_KEY environment variable not set. API will be accessible without authentication.")

# Create FastAPI app
app = FastAPI(
    title="NOI Data Extraction API (Enhanced + App Fees)",
    description="API for extracting detailed financial data (incl. Application Fees) from real estate documents",
    version="2.2.0" # Version bump
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for now, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define Pydantic Models for the MERGED Response Structure ---

class VacancyCreditLossDetails(BaseModel):
    physical_vacancy_loss: Optional[float] = 0.0
    concessions_free_rent: Optional[float] = 0.0
    bad_debt: Optional[float] = 0.0

class OtherIncomeDetails(BaseModel):
    recoveries: Optional[float] = 0.0
    parking_income: Optional[float] = 0.0
    laundry_income: Optional[float] = 0.0
    application_fees: Optional[float] = 0.0 # Added application_fees
    other_misc_income: Optional[float] = 0.0

class IncomeSummary(BaseModel):
    gross_potential_rent: Optional[float] = 0.0
    loss_to_lease: Optional[float] = 0.0
    vacancy_credit_loss_details: VacancyCreditLossDetails
    total_vacancy_credit_loss: Optional[float] = 0.0
    other_income_details: OtherIncomeDetails
    total_other_income: Optional[float] = 0.0
    effective_gross_income: Optional[float] = 0.0

class ExpenseDetails(BaseModel):
    property_taxes: Optional[float] = 0.0
    insurance: Optional[float] = 0.0
    property_management_fees: Optional[float] = 0.0
    repairs_maintenance: Optional[float] = 0.0
    utilities: Optional[float] = 0.0 # Could be float or dict if breakdown implemented
    payroll: Optional[float] = 0.0
    admin_office_costs: Optional[float] = 0.0
    marketing_advertising: Optional[float] = 0.0
    other_opex: Optional[float] = 0.0

class OperatingExpenses(BaseModel):
    expense_details: ExpenseDetails
    total_operating_expenses: Optional[float] = 0.0

class Reserves(BaseModel):
    # Allow None for reserves as they might not be present
    replacement_reserves: Optional[float] = None
    tenant_improvements: Optional[float] = None

class Financials(BaseModel):
    income_summary: IncomeSummary
    operating_expenses: OperatingExpenses
    net_operating_income: Optional[float] = 0.0
    reserves: Optional[Reserves] = None # Optional field

class DetailedMergedResponse(BaseModel): # Renamed for clarity
    document_type: str
    period: str
    financials: Financials
    filename: Optional[str] = None # Added filename for batch responses
    validation_warnings: Optional[List[str]] = None # Optional warnings field
    error: Optional[str] = None # For batch error reporting

# Helper function to validate API key (Unchanged)
def validate_api_key(api_key: Optional[str] = None, request: Optional[Request] = None) -> bool:
    """Validate API key from header or parameter."""
    env_api_key = os.environ.get("API_KEY")
    if not env_api_key:
        logger.warning("API_KEY environment variable not set. Allowing access (DEV ONLY).")
        return True # Allow if no key set in env (for local dev)

    provided_key = None
    source = "None"
    if api_key: # Check param first
        provided_key = api_key
        source = "Parameter"
    elif request: # Check headers if no param
        header_api_key = request.headers.get('x-api-key')
        if header_api_key:
             provided_key = header_api_key
             source = "x-api-key Header"
        else:
             auth_header = request.headers.get('Authorization')
             if auth_header and auth_header.startswith('Bearer '):
                  provided_key = auth_header.replace('Bearer ', '')
                  source = "Authorization Header"

    if provided_key:
        if provided_key == env_api_key:
            logger.info(f"API key validated via {source}")
            return True
        else:
             logger.warning(f"Invalid API key provided via {source}.")
             return False
    else:
        logger.warning("No API key provided in request.")
        return False


# Health check endpoint (Unchanged)
@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint."""
    user_agent = request.headers.get("user-agent", "").lower()
    if "render" in user_agent or "health" in user_agent:
         logger.info("Skipping auth for health check probe.")
         return {"status": "healthy", "version": "2.2.0"} # Updated version

    param_api_key = request.query_params.get('api_key')
    if not validate_api_key(param_api_key, request):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing API Key")

    return {"status": "healthy", "version": "2.2.0"} # Updated version

# Extract data from a single file (MODIFIED Response Model)
@app.post("/extract", response_model=DetailedMergedResponse) # Use updated model
async def extract_data(
    file: UploadFile = File(...),
    document_type: Optional[str] = Form(None),
    api_key: Optional[str] = Header(None, alias="x-api-key"),
    authorization: Optional[str] = Header(None),
    request: Request = None
):
    """Extract detailed financial data (incl. App Fees) from a single file."""
    auth_key = api_key
    if not auth_key and authorization and authorization.startswith('Bearer '):
        auth_key = authorization.replace('Bearer ', '')

    if not validate_api_key(auth_key, request):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing API Key")

    start_process_time = time.time()
    temp_file_path = None

    try:
        logger.info(f"Processing file: {file.filename} (type: {file.content_type})")
        # Securely create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or ".tmp")[1]) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
            logger.info(f"File saved temporarily to: {temp_file_path}")

        # --- Processing Steps ---
        preprocessed_data = preprocess_file(temp_file_path, file.content_type, file.filename)
        if isinstance(preprocessed_data, dict) and 'metadata' in preprocessed_data:
             preprocessed_data['metadata']['filename'] = file.filename

        if document_type:
            extraction_doc_type = map_noi_tool_to_extraction_type(document_type)
            period = classifier._extract_period_from_content(preprocessed_data) or classifier._extract_period_from_filename(file.filename)
        else:
            classification_result = classifier.classify(preprocessed_data)
            extraction_doc_type = classification_result['document_type']
            period = classification_result['period']
        logger.info(f"Document classified as '{extraction_doc_type}' for period '{period}'")

        extraction_result = extract_financial_data(preprocessed_data, extraction_doc_type, period)
        formatted_result = validate_and_format_data(extraction_result) # Returns final structure

        logger.info(f"Successfully processed {file.filename} in {time.time() - start_process_time:.2f}s")
        return formatted_result # Return the validated and formatted dict

    except HTTPException as http_exc:
         raise http_exc
    except FileNotFoundError as fnf_err:
         logger.error(f"File not found error: {str(fnf_err)}", exc_info=True)
         raise HTTPException(status_code=404, detail=f"File processing error: {str(fnf_err)}")
    except ValueError as val_err:
         logger.error(f"Value error during processing: {str(val_err)}", exc_info=True)
         raise HTTPException(status_code=400, detail=f"Invalid input: {str(val_err)}")
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error processing file: {file.filename}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Temporary file deleted: {temp_file_path}")
            except Exception as del_e:
                logger.error(f"Error deleting temporary file {temp_file_path}: {del_e}")


# Extract data from multiple files (batch processing) (MODIFIED Response Model)
@app.post("/extract-batch", response_model=List[DetailedMergedResponse]) # Use updated model
async def extract_batch(
    files: List[UploadFile] = File(...),
    document_types: Optional[str] = Form(None), # JSON string mapping filename to type
    api_key: Optional[str] = Header(None, alias="x-api-key"),
    authorization: Optional[str] = Header(None),
    request: Request = None
):
    """Extract detailed financial data (incl. App Fees) from multiple files."""
    auth_key = api_key
    if not auth_key and authorization and authorization.startswith('Bearer '):
        auth_key = authorization.replace('Bearer ', '')

    if not validate_api_key(auth_key, request):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing API Key")

    doc_types_map = {}
    if document_types:
        try:
            doc_types_map = json.loads(document_types)
            if not isinstance(doc_types_map, dict): raise ValueError("document_types JSON must be an object")
            logger.info(f"Received document types map: {doc_types_map}")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Invalid document_types JSON: {document_types} - Error: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid format for document_types: {e}")

    results = []
    batch_start_time = time.time()

    for file in files:
        file_start_time = time.time()
        temp_file_path = None
        file_result_data: Dict[str, Any] = {} # Holds data before Pydantic conversion
        extraction_doc_type = 'Unknown' # Default
        period = 'Unknown' # Default

        try:
            logger.info(f"Batch processing file: {file.filename} (type: {file.content_type})")
            doc_type_hint = doc_types_map.get(file.filename)

            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or ".tmp")[1]) as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_file_path = temp_file.name

            preprocessed_data = preprocess_file(temp_file_path, file.content_type, file.filename)
            if isinstance(preprocessed_data, dict) and 'metadata' in preprocessed_data:
                 preprocessed_data['metadata']['filename'] = file.filename

            if doc_type_hint:
                extraction_doc_type = map_noi_tool_to_extraction_type(doc_type_hint)
                period = classifier._extract_period_from_content(preprocessed_data) or classifier._extract_period_from_filename(file.filename)
            else:
                classification_result = classifier.classify(preprocessed_data)
                extraction_doc_type = classification_result['document_type']
                period = classification_result['period']

            extraction_result = extract_financial_data(preprocessed_data, extraction_doc_type, period)
            formatted_result = validate_and_format_data(extraction_result) # Returns final structure dict

            formatted_result['filename'] = file.filename # Add filename to the dict
            file_result_data = formatted_result
            logger.info(f"Successfully processed batch file {file.filename} in {time.time() - file_start_time:.2f}s")

        except Exception as e:
            logger.error(f"Error processing batch file {file.filename}: {str(e)}", exc_info=True)
            # Create error structure matching the response model as best as possible
            file_result_data = {
                 "filename": file.filename or "unknown_file",
                 "error": f"Failed to process: {str(e)}",
                 "document_type": extraction_doc_type, # Use last known type
                 "period": period or "Unknown", # Use last known period
                 "financials": Financials( # Provide default nested models for error case
                      income_summary=IncomeSummary(
                           vacancy_credit_loss_details=VacancyCreditLossDetails(),
                           other_income_details=OtherIncomeDetails()
                      ),
                      operating_expenses=OperatingExpenses(
                           expense_details=ExpenseDetails()
                      )
                 ).dict() # Convert default Pydantic models to dict for consistency here
            }
        finally:
             if temp_file_path and os.path.exists(temp_file_path):
                  try: os.unlink(temp_file_path)
                  except Exception as del_e: logger.error(f"Error deleting temp file {temp_file_path}: {del_e}")

        # Append the dictionary result (Pydantic handles conversion in the return)
        results.append(file_result_data)


    logger.info(f"Batch processing of {len(files)} files completed in {time.time() - batch_start_time:.2f}s")
    return results # FastAPI/Pydantic will validate each dict against DetailedMergedResponse


#################################################
# Main Entry Point
#################################################
if __name__ == "__main__":
    # Run API server using Uvicorn
    port = int(os.environ.get("PORT", 8000)) # Use PORT env var for Render/Heroku compatibility
    host = "0.0.0.0" # Listen on all available network interfaces

    # Consider adding --reload flag for local development if needed
    # Example: uvicorn.run("main_backend:app", host=host, port=port, reload=True)
    logger.info(f"Starting NOI Data Extraction API Server on {host}:{port}...")
    uvicorn.run("main_backend:app", host=host, port=port) # Reference the app instance directly

