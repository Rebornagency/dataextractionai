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
# Configure logging (Set level to DEBUG)
#################################################
logging.basicConfig(
    level=logging.DEBUG, # <-- Set to DEBUG to show detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Use a distinct logger name
logger = logging.getLogger('data_extraction_api_service_debug')

#################################################
# Preprocessing Module (Includes DEBUG LOGGING)
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
        Main method to preprocess a file. Ensures 'combined_text' is always present.
        """
        logger.debug(f"Preprocessing started for file: {filename or file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        effective_filename = filename or file_path
        _, ext = os.path.splitext(effective_filename)
        ext = ext.lower().lstrip('.')
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
        final_ext = detected_type_from_content or ext
        logger.info(f"Processing file '{effective_filename}'. Detected type: {final_ext} (from {'content-type' if detected_type_from_content else 'extension'}).")
        processor = self.supported_extensions.get(final_ext)
        if not processor and content_type and ('spreadsheet' in content_type or 'excel' in content_type.lower()):
            logger.warning(f"Extension '{final_ext}' not directly supported, but content-type '{content_type}' suggests Excel. Attempting Excel processing.")
            processor = self._process_excel
        elif not processor:
             try:
                 mime_type = magic.from_file(file_path, mime=True)
                 logger.info(f"python-magic detected MIME type: {mime_type}")
                 if 'pdf' in mime_type: processor = self._process_pdf
                 elif 'excel' in mime_type or 'spreadsheet' in mime_type: processor = self._process_excel
                 elif 'csv' in mime_type: processor = self._process_csv
                 elif 'text' in mime_type: processor = self._process_txt
                 else: raise ValueError(f"Unsupported file type: {final_ext} (MIME: {mime_type})")
             except Exception as magic_err:
                 logger.error(f"python-magic check failed: {magic_err}")
                 raise ValueError(f"Unsupported file type: {final_ext}")

        try:
            file_type_magic = magic.from_file(file_path, mime=True)
        except Exception as magic_err:
            logger.warning(f"python-magic failed to get MIME type: {magic_err}, using content_type if available.")
            file_type_magic = content_type or "unknown"
        file_size = os.path.getsize(file_path)
        logger.info(f"Processing with {processor.__name__}. File info: Type='{file_type_magic}', Size={file_size} bytes.")

        extracted_content = {} # Initialize
        try:
            extracted_content = processor(file_path) # Call the specific processor (_process_pdf, _process_excel, etc.)
            if 'combined_text' not in extracted_content:
                 logger.warning(f"Processor {processor.__name__} did not return 'combined_text'. Initializing empty.")
                 extracted_content['combined_text'] = ""
            elif not extracted_content['combined_text']:
                 logger.warning(f"Processor {processor.__name__} returned empty 'combined_text'.")

        except Exception as proc_err:
            logger.error(f"Error during processing with {processor.__name__}: {proc_err}", exc_info=True)
            extracted_content = {'combined_text': "", 'error': str(proc_err)}

        result = {
            'metadata': {
                'filename': filename or os.path.basename(file_path),
                'file_type_source': file_type_magic,
                'file_size_bytes': file_size,
                'detected_extension': final_ext,
                'original_content_type': content_type
            },
            'content': extracted_content
        }
        logger.debug(f"Preprocessing finished for {effective_filename}. Result keys: {list(result.keys())}, Content keys: {list(result.get('content', {}).keys())}")
        return result

    def _process_pdf(self, file_path: str) -> Dict[str, Any]:
        """Process PDF files using pdfplumber."""
        logger.debug(f"DEBUG: Starting _process_pdf for: {os.path.basename(file_path)}")
        result = {'text': [], 'tables': [], 'combined_text': ""}
        all_text_content = []
        try:
            with pdfplumber.open(file_path) as pdf:
                logger.debug(f"DEBUG: PDF has {len(pdf.pages)} pages.")
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    page_text = page.extract_text(x_tolerance=1, y_tolerance=3)
                    if page_text:
                        cleaned_page_text = self._clean_text(page_text)
                        result['text'].append({'page': page_num, 'content': cleaned_page_text})
                        all_text_content.append(cleaned_page_text)
                        logger.debug(f"DEBUG: Extracted text from PDF page {page_num}, length: {len(cleaned_page_text)}")
                    else:
                        logger.debug(f"DEBUG: No text extracted from PDF page {page_num}")

                    try:
                        tables = page.extract_tables(table_settings={
                            "vertical_strategy": "lines", "horizontal_strategy": "lines",
                            "snap_tolerance": 5, "join_tolerance": 5,
                        })
                        logger.debug(f"DEBUG: Found {len(tables)} potential tables on PDF page {page_num}")
                        for j, table in enumerate(tables):
                            if table:
                                cleaned_table = [[str(cell) if cell is not None else '' for cell in row] for row in table]
                                if not cleaned_table: continue
                                df = pd.DataFrame(cleaned_table)
                                header_index = self._find_header_row(df)
                                if header_index is not None:
                                    df.columns = df.iloc[header_index]
                                    df = df.iloc[header_index + 1:].reset_index(drop=True)
                                df = df.replace({np.nan: None})
                                result['tables'].append({
                                    'page': page_num, 'table_index': j, 'data': df.to_dict(orient='records')
                                })
                    except Exception as table_err:
                        logger.warning(f"Could not extract tables from page {page_num}: {table_err}")

        except Exception as e:
            logger.error(f"Error processing PDF file '{os.path.basename(file_path)}': {str(e)}")
        finally:
            result['combined_text'] = "\n\n--- Page Break ---\n\n".join(all_text_content)
            logger.info(f"Finished PDF processing. Extracted {len(result['text'])} text pages, {len(result['tables'])} tables.")
            logger.debug(f"DEBUG: Final combined_text length (PDF): {len(result.get('combined_text', ''))}")
        return result

    def _process_excel(self, file_path: str) -> Dict[str, Any]:
        """Process Excel files using pandas."""
        logger.debug(f"DEBUG: Starting _process_excel for: {os.path.basename(file_path)}")
        result = {'sheets': [], 'text_representation': [], 'combined_text': ""}
        all_text_repr: List[str] = []

        try:
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names
            logger.debug(f"DEBUG: Excel sheets found: {sheet_names}")

            for sheet_name in sheet_names:
                logger.debug(f"DEBUG: Processing sheet '{sheet_name}'")
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, keep_default_na=True)
                    df.dropna(axis=0, how='all', inplace=True)
                    df.dropna(axis=1, how='all', inplace=True)
                    df.reset_index(drop=True, inplace=True)

                    if df.empty:
                         logger.debug(f"DEBUG: Sheet '{sheet_name}' is empty after cleaning, skipping.")
                         continue

                    header_row_index = self._find_header_row(df)
                    if header_row_index is not None:
                         logger.debug(f"DEBUG: Detected header row {header_row_index} in sheet '{sheet_name}'.")
                         df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row_index, keep_default_na=True)
                         df.dropna(axis=0, how='all', inplace=True)
                         df.dropna(axis=1, how='all', inplace=True)
                    else:
                         logger.warning(f"No clear header found in sheet '{sheet_name}', using default headers.")
                         df.columns = [f'Column_{k}' for k in range(len(df.columns))]

                    df = df.replace({np.nan: None, pd.NaT: None})
                    df_str = df.astype(str).replace({'None': ''})

                    result['sheets'].append({'name': sheet_name, 'data': df.to_dict(orient='records')})

                    text_rep = f"--- Sheet: {sheet_name} ---\n"
                    try:
                        from tabulate import tabulate
                        text_rep += tabulate(df_str, headers='keys', tablefmt='grid', showindex=False)
                    except ImportError:
                        text_rep += df_str.to_string(index=False, na_rep='')
                    all_text_repr.append(text_rep)
                    logger.debug(f"DEBUG: Added text representation for sheet '{sheet_name}'. Length: {len(text_rep)}")

                except Exception as sheet_error:
                     logger.warning(f"Could not process sheet '{sheet_name}': {sheet_error}", exc_info=True)

        except Exception as e:
            logger.error(f"Error processing Excel file '{os.path.basename(file_path)}': {str(e)}")
        finally:
            result['text_representation'] = all_text_repr
            result['combined_text'] = "\n\n".join(all_text_repr)
            logger.info(f"Finished Excel processing. Processed {len(result['sheets'])} sheets.")
            logger.debug(f"DEBUG: Final combined_text length (Excel): {len(result.get('combined_text', ''))}")
        return result

    def _process_csv(self, file_path: str) -> Dict[str, Any]:
        """Process CSV files using pandas."""
        logger.debug(f"DEBUG: Starting _process_csv for: {os.path.basename(file_path)}")
        result = {'data': [], 'text_representation': "", 'combined_text': ""}
        text_rep = ""

        try:
            encoding = self._detect_encoding(file_path)
            logger.debug(f"DEBUG: Reading CSV with encoding: {encoding}")
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False, on_bad_lines='skip')
            df.dropna(axis=0, how='all', inplace=True)
            df.dropna(axis=1, how='all', inplace=True)

            if not df.empty:
                df = df.replace({np.nan: None, pd.NaT: None})
                result['data'] = df.to_dict(orient='records')
                df_str = df.astype(str).replace({'None': ''})
                try:
                    from tabulate import tabulate
                    text_rep = tabulate(df_str, headers='keys', tablefmt='grid', showindex=False)
                except ImportError:
                    text_rep = df_str.to_string(index=False, na_rep='')
                logger.info(f"Finished CSV processing. Extracted {len(df)} rows.")
            else:
                logger.warning(f"CSV file '{os.path.basename(file_path)}' is empty or contains only invalid lines.")

        except Exception as e:
            logger.error(f"Error processing CSV file '{os.path.basename(file_path)}': {str(e)}")
        finally:
            result['text_representation'] = text_rep
            result['combined_text'] = text_rep
            logger.debug(f"DEBUG: Final combined_text length (CSV): {len(result.get('combined_text', ''))}")
        return result

    def _process_txt(self, file_path: str) -> Dict[str, Any]:
        """Process TXT files."""
        logger.debug(f"DEBUG: Starting _process_txt for: {os.path.basename(file_path)}")
        result = {'text': "", 'combined_text': ""}
        cleaned_text = ""
        try:
            encoding = self._detect_encoding(file_path)
            logger.debug(f"DEBUG: Reading TXT with encoding: {encoding}")
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                text = f.read()
            cleaned_text = self._clean_text(text)
            logger.info(f"Finished TXT processing.")
        except Exception as e:
            logger.error(f"Error processing TXT file '{os.path.basename(file_path)}': {str(e)}")
        finally:
            result['text'] = cleaned_text
            result['combined_text'] = cleaned_text
            logger.debug(f"DEBUG: Final combined_text length (TXT): {len(result.get('combined_text', ''))}")
        return result

    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding using chardet."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(4096)
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'
                logger.debug(f"DEBUG: Detected encoding: {encoding} (confidence: {result['confidence']:.2f})")
                return encoding
        except Exception as e:
            logger.warning(f"Error detecting encoding: {e}. Falling back to utf-8.")
            return 'utf-8'

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        if not text:
            return ""
        # Replace multiple spaces with single space
        cleaned = re.sub(r'\s+', ' ', text)
        # Replace multiple newlines with single newline
        cleaned = re.sub(r'\n+', '\n', cleaned)
        # Remove control characters except newlines and tabs
        cleaned = re.sub(r'[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)
        return cleaned.strip()

    def _find_header_row(self, df: pd.DataFrame) -> Optional[int]:
        """Find the most likely header row in a dataframe."""
        if df.empty or len(df) < 2:
            return None

        # Check first 10 rows or all rows if fewer
        max_check = min(10, len(df))
        best_score = -1
        best_row = None

        for i in range(max_check):
            row = df.iloc[i]
            # Skip rows with too many NaNs
            if row.isna().mean() > 0.5:
                continue

            # Convert row values to strings and check characteristics
            str_values = [str(v).strip() if v is not None else '' for v in row.values]
            
            # Count characteristics that suggest a header
            uppercase_count = sum(1 for v in str_values if v and v.isupper())
            title_case_count = sum(1 for v in str_values if v and v.istitle())
            short_value_count = sum(1 for v in str_values if 1 < len(v) < 25)  # Headers are usually not too long
            non_numeric_count = sum(1 for v in str_values if v and not v.replace('.', '').replace(',', '').replace('-', '').isdigit())
            
            # Calculate score based on these characteristics
            score = (uppercase_count * 0.5 + title_case_count * 0.3 + short_value_count * 0.1 + non_numeric_count * 0.1) / max(1, len(str_values))
            
            if score > best_score:
                best_score = score
                best_row = i

        # Only return a row if it has a reasonable score
        return best_row if best_score > 0.3 else None

#################################################
# Document Classification Module
#################################################
class DocumentClassifier:
    """Class for classifying financial documents and extracting time periods."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the document classifier."""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key: logger.warning("OpenAI API key not set.")
        self.client = OpenAI(api_key=self.api_key)
        self.document_types = ["Actual Income Statement", "Budget", "Prior Year Actual", "Unknown"]
        self.document_type_mapping = {"current_month_actuals": "Actual Income Statement", "prior_month_actuals": "Actual Income Statement", "current_month_budget": "Budget", "prior_year_actuals": "Prior Year Actual"}

    def classify(self, text_or_data: Union[str, Dict[str, Any]], known_document_type: Optional[str] = None) -> Dict[str, Any]:
        """Classify document type and extract time period."""
        logger.info("Classifying document and extracting time period")
        if known_document_type:
            logger.info(f"Using known document type: {known_document_type}")
            if known_document_type in self.document_type_mapping:
                doc_type = self.document_type_mapping[known_document_type]
                logger.info(f"Mapped known document type '{known_document_type}' to '{doc_type}'")
                period = self._extract_period_from_content(text_or_data)
                return {'document_type': doc_type, 'period': period, 'method': 'known_type'}
            else:
                logger.warning(f"Unknown document type mapping for '{known_document_type}', falling back to extraction")
        extracted_text = self._extract_text_from_input(text_or_data)
        filename = None
        if isinstance(text_or_data, dict) and 'metadata' in text_or_data and 'filename' in text_or_data['metadata']:
            filename = text_or_data['metadata']['filename']
            logger.info(f"Found filename in metadata: {filename}")
        doc_type_from_filename = self._determine_type_from_filename(filename) if filename else None
        if doc_type_from_filename:
            logger.info(f"Determined document type from filename: {doc_type_from_filename}")
            period = self._extract_period_from_content(extracted_text) or self._extract_period_from_filename(filename)
            return {'document_type': doc_type_from_filename, 'period': period, 'method': 'filename'}
        rule_based_result = self._rule_based_classification(extracted_text)
        if rule_based_result.get('confidence', 0) > 0.7:
            logger.info(f"Rule-based classification successful: {rule_based_result}")
            return {'document_type': rule_based_result['document_type'], 'period': rule_based_result['period'], 'method': 'rule_based'}
        gpt_result = self._gpt_classification(extracted_text)
        logger.info(f"GPT classification result: {gpt_result}")
        return {'document_type': gpt_result['document_type'], 'period': gpt_result['period'], 'method': 'gpt'}

    def _extract_text_from_input(self, text_or_data: Union[str, Dict[str, Any]]) -> str:
        """Extract text content from input which could be string or dictionary."""
        if isinstance(text_or_data, dict):
            if 'combined_text' in text_or_data: return text_or_data['combined_text']
            elif 'text' in text_or_data:
                if isinstance(text_or_data['text'], str): return text_or_data['text']
                elif isinstance(text_or_data['text'], list) and len(text_or_data['text']) > 0:
                    page_contents = []
                    for page in text_or_data['text']:
                        if isinstance(page, dict) and 'content' in page and page['content']: page_contents.append(page['content'])
                        elif isinstance(page, str): page_contents.append(page)
                    return "\n\n".join(page_contents)
            elif 'content' in text_or_data and isinstance(text_or_data['content'], str): return text_or_data['content']
            elif 'data' in text_or_data and isinstance(text_or_data['data'], str): return text_or_data['data']
            elif 'sheets' in text_or_data and isinstance(text_or_data['sheets'], list):
                 if 'text_representation' in text_or_data and isinstance(text_or_data['text_representation'], list): return "\n\n".join(text_or_data['text_representation'])
            logger.warning("Could not find specific text field in dictionary, using JSON string representation")
            try: return json.dumps(text_or_data, indent=2, default=str, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Error converting dict to JSON string: {e}")
                return str(text_or_data)[:2000]
        else:
            if isinstance(text_or_data, str): return text_or_data
            else:
                logger.warning(f"Converted non-string input to string: {type(text_or_data)}")
                return str(text_or_data)

    def _extract_period_from_content(self, text_or_data: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extract period information from document content."""
        text = self._extract_text_from_input(text_or_data)
        if not text: return None
        months_full = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        months_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        months_pattern = '|'.join(months_full + months_abbr)
        month_year_pattern = rf'(?:For the (?:Month|Period) Ended\s+)?({months_pattern})[\s.,_-]*(\d{{1,2}}[\s.,_-]*)?(\d{{4}})'
        quarter_words = ['First', 'Second', 'Third', 'Fourth']
        quarter_pattern = rf'(?:(Q[1-4])|({"|".join(quarter_words)})\s+Quarter)[\s.,_-]*(\d{{4}})|(?:Quarter Ended\s+(?:{months_pattern})\s+\d{{1,2}}[\s.,_-]*)(\d{{4}})'
        year_pattern = r'(?:For the Year Ended|FY|Fiscal Year|Calendar Year)[\s.,_-]*(\d{4})'
        month_year_match = re.search(month_year_pattern, text, re.IGNORECASE)
        if month_year_match:
            month = month_year_match.group(1)
            year = month_year_match.group(3)
            for i, abbr in enumerate(months_abbr):
                if month.lower().startswith(abbr.lower()):
                    month = months_full[i]
                    break
            return f"{month} {year}"
        quarter_match = re.search(quarter_pattern, text, re.IGNORECASE)
        if quarter_match:
            if quarter_match.group(1): return f"{quarter_match.group(1)} {quarter_match.group(3)}"
            elif quarter_match.group(2): return f"{quarter_match.group(2)} Quarter {quarter_match.group(3)}"
            elif quarter_match.group(4): return f"Quarter Ended {quarter_match.group(4)}"
        year_match = re.search(year_pattern, text, re.IGNORECASE)
        if year_match: return f"Year {year_match.group(1)}"
        if "prior year" in text.lower() or "previous year" in text.lower():
            year_only_match = re.search(r'\b(20\d{2})\b', text)
            if year_only_match: return f"Year {year_only_match.group(1)}"
        return None

    def _extract_period_from_filename(self, filename: str) -> Optional[str]:
        """Extract period information from filename."""
        if not filename: return None
        name_part=os.path.splitext(filename)[0]; name_part=re.sub(r'[_\-\.]+',' ',name_part); months_full=['January','February','March','April','May','June','July','August','September','October','November','December']; months_abbr=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']; months_pattern='|'.join(months_full+months_abbr); year_pattern=r'(20\d{2})'; month_year_match=re.search(rf'({months_pattern})\s+{year_pattern}',name_part,re.IGNORECASE);
        if month_year_match: return f"{self._standardize_month(month_year_match.group(1),months_full,months_abbr)} {month_year_match.group(2)}"; month_year_match2=re.search(rf'{year_pattern}\s+({months_pattern})',name_part,re.IGNORECASE);
        if month_year_match2: return f"{self._standardize_month(month_year_match2.group(2),months_full,months_abbr)} {month_year_match2.group(1)}"; quarter_match=re.search(rf'(Q[1-4])\s*{year_pattern}',name_part,re.IGNORECASE);
        if quarter_match: return f"{quarter_match.group(1)} {quarter_match.group(2)}"; quarter_match_rev=re.search(rf'{year_pattern}\s*(Q[1-4])',name_part,re.IGNORECASE);
        if quarter_match_rev: return f"{quarter_match_rev.group(2)} {quarter_match_rev.group(1)}"; year_match=re.search(year_pattern,name_part);
        if year_match and any(kw in name_part.lower() for kw in ['budget','actual','year','report']): return f"Year {year_match.group(1)}"; return None

    def _rule_based_classification(self, text: str) -> Dict[str, Any]:
        """Attempt rule-based classification."""
        if not text: return {'document_type': 'Unknown', 'period': None, 'confidence': 0.0}
        text_lower = text.lower()
        confidence = 0.0
        doc_type = 'Unknown'
        period = None
        # Check for budget indicators
        budget_indicators = ['budget', 'projected', 'forecast', 'plan', 'target']
        budget_score = sum(1 for indicator in budget_indicators if indicator in text_lower)
        # Check for actual indicators
        actual_indicators = ['actual', 'income statement', 'profit and loss', 'p&l', 'statement of operations']
        actual_score = sum(1 for indicator in actual_indicators if indicator in text_lower)
        # Check for prior year indicators
        prior_year_indicators = ['prior year', 'previous year', 'last year', 'year over year', 'yoy']
        prior_year_score = sum(1 for indicator in prior_year_indicators if indicator in text_lower)
        # Determine document type based on scores
        if budget_score > actual_score and budget_score > prior_year_score:
            doc_type = 'Budget'
            confidence = min(0.5 + budget_score * 0.1, 0.9)
        elif prior_year_score > 0:
            doc_type = 'Prior Year Actual'
            confidence = min(0.5 + prior_year_score * 0.1, 0.9)
        elif actual_score > 0:
            doc_type = 'Actual Income Statement'
            confidence = min(0.5 + actual_score * 0.1, 0.9)
        # Extract period
        period = self._extract_period_from_content(text)
        if period: confidence = min(confidence + 0.1, 0.95)
        return {'document_type': doc_type, 'period': period, 'confidence': confidence}

    def _determine_type_from_filename(self, filename: str) -> Optional[str]:
        """Determine document type from filename."""
        if not filename: return None
        filename_lower = filename.lower()
        if 'budget' in filename_lower or 'forecast' in filename_lower or 'plan' in filename_lower:
            return 'Budget'
        elif 'prior year' in filename_lower or 'previous year' in filename_lower or 'last year' in filename_lower:
            return 'Prior Year Actual'
        elif 'actual' in filename_lower or 'income statement' in filename_lower or 'p&l' in filename_lower:
            return 'Actual Income Statement'
        return None

    def _standardize_month(self, month_str: str, months_full: List[str], months_abbr: List[str]) -> str:
        """Standardize month name to full name."""
        month_lower = month_str.lower()
        for i, month in enumerate(months_full):
            if month_lower == month.lower():
                return month
        for i, abbr in enumerate(months_abbr):
            if month_lower.startswith(abbr.lower()):
                return months_full[i]
        return month_str

    def _gpt_classification(self, text: str) -> Dict[str, Any]:
        """Use GPT to classify document type and extract period."""
        if not self.api_key:
            logger.warning("OpenAI API key not set, skipping GPT classification")
            return {'document_type': 'Unknown', 'period': None, 'method': 'gpt_fallback'}
        # Truncate text to avoid token limits
        truncated_text = text[:4000]
        try:
            prompt = f"""Analyze the following financial document text and determine:
1. Document Type: Is this an "Actual Income Statement", "Budget", "Prior Year Actual", or "Unknown"?
2. Time Period: What month/quarter/year does this document cover?

Extract this information in JSON format with keys "document_type" and "period".

Document Text:
{truncated_text}

JSON Response:"""
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a financial document classifier that extracts document type and time period information."}, {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150
            )
            result_text = response.choices[0].message.content
            # Extract JSON from response
            json_match = re.search(r'({.*})', result_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                    # Validate and clean result
                    if 'document_type' not in result or not result['document_type']:
                        result['document_type'] = 'Unknown'
                    if 'period' not in result or not result['period']:
                        result['period'] = None
                    return result
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse GPT response as JSON: {result_text}")
            logger.warning(f"GPT response did not contain valid JSON: {result_text}")
            return {'document_type': 'Unknown', 'period': None, 'method': 'gpt_parse_error'}
        except Exception as e:
            logger.error(f"Error in GPT classification: {str(e)}")
            return {'document_type': 'Unknown', 'period': None, 'method': 'gpt_error'}

#################################################
# GPT Data Extraction Module
#################################################
class GPTDataExtractor:
    """Class for extracting financial data using GPT."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the GPT data extractor."""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key: logger.warning("OpenAI API key not set.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4"  # Default to GPT-4 for better extraction
        self.max_tokens = 1000
        self.temperature = 0.0
        self.retry_count = 3
        self.retry_delay = 2  # seconds

    def extract_data(self, text_or_data: Union[str, Dict[str, Any]], document_type: str, period: Optional[str] = None) -> Dict[str, Any]:
        """Extract financial data from document."""
        logger.info(f"Extracting data from document type: {document_type}, period: {period}")
        extracted_text = self._extract_text_from_input(text_or_data)
        if not extracted_text:
            logger.warning("No text content to extract data from")
            return self._empty_result(document_type, period)
        # Truncate text to avoid token limits
        truncated_text = self._truncate_text(extracted_text)
        # Prepare extraction prompt
        prompt = self._prepare_extraction_prompt(truncated_text, document_type, period)
        # Extract data using GPT
        extraction_result = self._extract_with_gpt(prompt)
        if not extraction_result:
            logger.warning("GPT extraction failed")
            return self._empty_result(document_type, period)
        # Add document type and period to result
        extraction_result['document_type'] = document_type
        extraction_result['period'] = period
        return extraction_result

    def _extract_text_from_input(self, text_or_data: Union[str, Dict[str, Any]]) -> str:
        """Extract text content from input which could be string or dictionary."""
        if isinstance(text_or_data, dict):
            # If it's a preprocessed result from FilePreprocessor
            if 'content' in text_or_data and isinstance(text_or_data['content'], dict):
                content = text_or_data['content']
                if 'combined_text' in content and content['combined_text']:
                    return content['combined_text']
                elif 'text' in content:
                    if isinstance(content['text'], str):
                        return content['text']
                    elif isinstance(content['text'], list):
                        return "\n\n".join([page['content'] if isinstance(page, dict) and 'content' in page else str(page) for page in content['text'] if page])
            # If it's a direct text field
            elif 'combined_text' in text_or_data and text_or_data['combined_text']:
                return text_or_data['combined_text']
            elif 'text' in text_or_data:
                if isinstance(text_or_data['text'], str):
                    return text_or_data['text']
                elif isinstance(text_or_data['text'], list):
                    return "\n\n".join([page['content'] if isinstance(page, dict) and 'content' in page else str(page) for page in text_or_data['text'] if page])
            # Try to get text from tables
            elif 'tables' in text_or_data and isinstance(text_or_data['tables'], list):
                table_texts = []
                for table in text_or_data['tables']:
                    if isinstance(table, dict) and 'data' in table:
                        try:
                            table_texts.append(json.dumps(table['data'], indent=2))
                        except:
                            pass
                if table_texts:
                    return "\n\n".join(table_texts)
            # If it's a metadata wrapper
            elif 'metadata' in text_or_data and 'content' in text_or_data:
                return self._extract_text_from_input(text_or_data['content'])
            # Last resort: convert to JSON
            logger.warning("Could not find specific text field in dictionary, using JSON string representation")
            try:
                return json.dumps(text_or_data, indent=2, default=str)[:10000]
            except:
                return str(text_or_data)[:10000]
        else:
            # If it's already a string
            return str(text_or_data)

    def _truncate_text(self, text: str, max_chars: int = 15000) -> str:
        """Truncate text to avoid token limits."""
        if len(text) <= max_chars:
            return text
        # Try to truncate at paragraph boundaries
        paragraphs = text.split("\n\n")
        result = ""
        for para in paragraphs:
            if len(result) + len(para) + 2 <= max_chars:
                result += para + "\n\n"
            else:
                break
        # If we couldn't get enough text with paragraph truncation, just cut at max_chars
        if len(result) < max_chars / 2:
            result = text[:max_chars]
        return result

    def _prepare_extraction_prompt(self, text: str, document_type: str, period: Optional[str] = None) -> str:
        """Prepare extraction prompt based on document type."""
        period_str = f" for period {period}" if period else ""
        prompt = f"""Extract detailed financial data from this {document_type}{period_str}. 

Focus on extracting these key financial metrics:
1. Gross Potential Rent (GPR)
2. Vacancy & Credit Loss details:
   - Physical Vacancy Loss
   - Concessions/Free Rent
   - Bad Debt
   - Total Vacancy & Credit Loss
3. Other Income details:
   - Recoveries
   - Parking Income
   - Laundry Income
   - Application Fees
   - Other Miscellaneous Income
   - Total Other Income
4. Effective Gross Income (EGI)
5. Operating Expenses details:
   - Property Taxes
   - Insurance
   - Property Management Fees
   - Repairs & Maintenance
   - Utilities
   - Payroll
   - Administrative/Office Costs
   - Marketing/Advertising
   - Other Operating Expenses
   - Total Operating Expenses
6. Net Operating Income (NOI)
7. Reserves (if available):
   - Replacement Reserves
   - Tenant Improvements

Return the data as a JSON object with numeric values (use 0 for missing values, not null or empty strings).

Document Text:
{text}

JSON Response:"""
        return prompt

    def _extract_with_gpt(self, prompt: str) -> Dict[str, Any]:
        """Extract data using GPT with retry logic."""
        for attempt in range(self.retry_count):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a financial data extraction assistant that extracts structured data from financial documents."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                result_text = response.choices[0].message.content
                # Extract JSON from response
                json_match = re.search(r'({.*})', result_text, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                        # Convert any string numbers to float
                        self._convert_string_numbers_to_float(result)
                        return result
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse GPT response as JSON: {e}")
                        logger.debug(f"Raw response: {result_text}")
                else:
                    logger.warning(f"GPT response did not contain valid JSON")
                    logger.debug(f"Raw response: {result_text}")
            except RateLimitError:
                logger.warning(f"Rate limit exceeded, retrying in {self.retry_delay} seconds (attempt {attempt+1}/{self.retry_count})")
                time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
            except APITimeoutError:
                logger.warning(f"API timeout, retrying in {self.retry_delay} seconds (attempt {attempt+1}/{self.retry_count})")
                time.sleep(self.retry_delay)
            except APIError as e:
                logger.error(f"API error: {e}")
                break
            except Exception as e:
                logger.error(f"Error in GPT extraction: {e}")
                break
        return {}

    def _convert_string_numbers_to_float(self, data: Any) -> None:
        """Recursively convert string numbers to float in a nested structure."""
        if isinstance(data, dict):
            for key, value in list(data.items()):
                if isinstance(value, str) and self._is_numeric(value):
                    data[key] = self._parse_numeric(value)
                elif isinstance(value, (dict, list)):
                    self._convert_string_numbers_to_float(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str) and self._is_numeric(item):
                    data[i] = self._parse_numeric(item)
                elif isinstance(item, (dict, list)):
                    self._convert_string_numbers_to_float(item)

    def _is_numeric(self, value: str) -> bool:
        """Check if a string represents a numeric value."""
        # Remove currency symbols, commas, and spaces
        cleaned = re.sub(r'[$,\s]', '', value)
        # Check if it's a number (including negative and decimal)
        return re.match(r'^-?\d+(\.\d+)?$', cleaned) is not None

    def _parse_numeric(self, value: str) -> float:
        """Parse a string as a numeric value."""
        # Remove currency symbols, commas, and spaces
        cleaned = re.sub(r'[$,\s]', '', value)
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    def _empty_result(self, document_type: str, period: Optional[str] = None) -> Dict[str, Any]:
        """Return an empty result structure."""
        return {
            "document_type": document_type,
            "period": period or "Unknown",
            "gross_potential_rent": 0.0,
            "physical_vacancy_loss": 0.0,
            "concessions_free_rent": 0.0,
            "bad_debt": 0.0,
            "total_vacancy_credit_loss": 0.0,
            "recoveries": 0.0,
            "parking_income": 0.0,
            "laundry_income": 0.0,
            "application_fees": 0.0,
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
            "replacement_reserves": None,
            "tenant_improvements": None
        }

#################################################
# Validation and Formatting Module
#################################################
class ValidationFormatter:
    """Class for validating and formatting extracted data."""

    def __init__(self):
        """Initialize the validation formatter."""
        pass

    def validate_and_format(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Validate and format extracted data."""
        # Make a copy to avoid modifying the original
        formatted_data = data.copy()
        warnings = []
        
        # Ensure all required fields are present with default values
        self._ensure_required_fields(formatted_data)
        
        # Validate and clean numeric values
        self._validate_numeric_values(formatted_data)
        
        # Perform validation checks
        warnings.extend(self._validate_vacancy_sum(formatted_data))
        warnings.extend(self._validate_other_income_sum(formatted_data))
        warnings.extend(self._validate_egi(formatted_data))
        warnings.extend(self._validate_total_opex_sum(formatted_data))
        warnings.extend(self._validate_noi_from_egi(formatted_data))
        
        # Format the output
        formatted_output = self._format_output(formatted_data)
        
        return formatted_output, warnings

    def _ensure_required_fields(self, data: Dict[str, Any]) -> None:
        """Ensure all required fields are present with default values."""
        required_fields = {
            "document_type": "Unknown",
            "period": "Unknown",
            "gross_potential_rent": 0.0,
            "physical_vacancy_loss": 0.0,
            "concessions_free_rent": 0.0,
            "bad_debt": 0.0,
            "total_vacancy_credit_loss": 0.0,
            "recoveries": 0.0,
            "parking_income": 0.0,
            "laundry_income": 0.0,
            "application_fees": 0.0,
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
            "net_operating_income": 0.0
        }
        
        for field, default_value in required_fields.items():
            if field not in data or data[field] is None:
                data[field] = default_value

    def _validate_numeric_values(self, data: Dict[str, Any]) -> None:
        """Validate and clean numeric values."""
        numeric_fields = [
            "gross_potential_rent",
            "physical_vacancy_loss",
            "concessions_free_rent",
            "bad_debt",
            "total_vacancy_credit_loss",
            "recoveries",
            "parking_income",
            "laundry_income",
            "application_fees",
            "other_misc_income",
            "total_other_income",
            "effective_gross_income",
            "property_taxes",
            "insurance",
            "property_management_fees",
            "repairs_maintenance",
            "utilities",
            "payroll",
            "admin_office_costs",
            "marketing_advertising",
            "other_opex",
            "total_operating_expenses",
            "net_operating_income",
            "replacement_reserves",
            "tenant_improvements"
        ]
        
        for field in numeric_fields:
            if field in data:
                # Skip None values for optional fields
                if field in ["replacement_reserves", "tenant_improvements"] and data[field] is None:
                    continue
                
                # Convert to float and handle invalid values
                try:
                    if isinstance(data[field], str):
                        # Remove currency symbols, commas, and spaces
                        cleaned = re.sub(r'[$,\s]', '', data[field])
                        data[field] = float(cleaned)
                    else:
                        data[field] = float(data[field])
                    
                    # Check for NaN or Infinity
                    if math.isnan(data[field]) or math.isinf(data[field]):
                        data[field] = 0.0
                except (ValueError, TypeError):
                    data[field] = 0.0

    def _clean_value_for_validation(self, data: Dict[str, Any], field: str) -> float:
        """Clean a value for validation purposes."""
        if field not in data or data[field] is None:
            return 0.0
        try:
            value = float(data[field])
            return 0.0 if math.isnan(value) or math.isinf(value) else value
        except (ValueError, TypeError):
            return 0.0

    def _validate_vacancy_sum(self, data: Dict[str, Any]) -> List[str]:
        """Validate total_vacancy_credit_loss against its components"""
        warnings = []
        tolerance = 1.0
        components = [self._clean_value_for_validation(data, f) for f in ['physical_vacancy_loss', 'concessions_free_rent', 'bad_debt']]
        calculated_sum = sum(components)
        reported_total = self._clean_value_for_validation(data, 'total_vacancy_credit_loss')
        if abs(calculated_sum - reported_total) > tolerance:
            warnings.append(f"Validation Check: Total Vacancy ({reported_total:.2f}) vs sum ({calculated_sum:.2f}).")
        return warnings

    def _validate_other_income_sum(self, data: Dict[str, Any]) -> List[str]:
        """Validate total_other_income against its components (incl. application_fees)"""
        warnings = []
        tolerance = 1.0
        components = [self._clean_value_for_validation(data, f) for f in ['recoveries', 'parking_income', 'laundry_income', 'application_fees', 'other_misc_income']]
        calculated_sum = sum(components)
        reported_total = self._clean_value_for_validation(data, 'total_other_income')
        if abs(calculated_sum - reported_total) > tolerance:
            warnings.append(f"Validation Check: Total Other Income ({reported_total:.2f}) vs sum ({calculated_sum:.2f}).")
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
             warnings.append(f"Validation Check: EGI ({reported_egi:.2f}) vs calc ({calculated_egi:.2f}).")
        return warnings

    def _validate_total_opex_sum(self, data: Dict[str, Any]) -> List[str]:
        """Validate total_operating_expenses against its components"""
        warnings = []
        tolerance = 1.0
        opex_components = ['property_taxes', 'insurance', 'property_management_fees', 'repairs_maintenance', 'utilities', 'payroll', 'admin_office_costs', 'marketing_advertising', 'other_opex']
        calculated_sum = sum(self._clean_value_for_validation(data, field) for field in opex_components)
        reported_total = self._clean_value_for_validation(data, 'total_operating_expenses')
        if abs(calculated_sum - reported_total) > tolerance:
            warnings.append(f"Validation Check: Total OpEx ({reported_total:.2f}) vs sum ({calculated_sum:.2f}).")
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
            warnings.append(f"Validation Check: NOI ({reported_noi:.2f}) vs calc ({calculated_noi:.2f}).")
        return warnings

    def _format_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Formats the data into the final nested JSON structure."""
        output = {"document_type": data.get('document_type', 'Unknown'), "period": data.get('period', 'Unknown'), "financials": {"income_summary": {"gross_potential_rent": data.get('gross_potential_rent', 0.0), "loss_to_lease": data.get('loss_to_lease', 0.0), "vacancy_credit_loss_details": {"physical_vacancy_loss": data.get('physical_vacancy_loss', 0.0), "concessions_free_rent": data.get('concessions_free_rent', 0.0), "bad_debt": data.get('bad_debt', 0.0)}, "total_vacancy_credit_loss": data.get('total_vacancy_credit_loss', 0.0), "other_income_details": {"recoveries": data.get('recoveries', 0.0), "parking_income": data.get('parking_income', 0.0), "laundry_income": data.get('laundry_income', 0.0), "application_fees": data.get('application_fees', 0.0), "other_misc_income": data.get('other_misc_income', 0.0)}, "total_other_income": data.get('total_other_income', 0.0), "effective_gross_income": data.get('effective_gross_income', 0.0)}, "operating_expenses": {"expense_details": {"property_taxes": data.get('property_taxes', 0.0), "insurance": data.get('insurance', 0.0), "property_management_fees": data.get('property_management_fees', 0.0), "repairs_maintenance": data.get('repairs_maintenance', 0.0), "utilities": data.get('utilities', 0.0), "payroll": data.get('payroll', 0.0), "admin_office_costs": data.get('admin_office_costs', 0.0), "marketing_advertising": data.get('marketing_advertising', 0.0), "other_opex": data.get('other_opex', 0.0)}, "total_operating_expenses": data.get('total_operating_expenses', 0.0)}, "net_operating_income": data.get('net_operating_income', 0.0), "reserves": {"replacement_reserves": data.get('replacement_reserves'), "tenant_improvements": data.get('tenant_improvements')}}}
        return output

def validate_and_format_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to validate and format extracted detailed financial data."""
    validator = ValidationFormatter()
    formatted_data, _ = validator.validate_and_format(data)
    return formatted_data

#################################################
# API Server Module (Unchanged)
#################################################
API_KEY = os.environ.get("API_KEY");
if not API_KEY: logger.warning("API_KEY env var not set.")
# Define app instance here so it's accessible globally in the script
app = FastAPI(title="NOI Data Extraction API (v2.2.4)", description="API for extracting detailed financial data", version="2.2.4") # Version bump
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
class VacancyCreditLossDetails(BaseModel): physical_vacancy_loss: Optional[float]=0.0; concessions_free_rent: Optional[float]=0.0; bad_debt: Optional[float]=0.0
class OtherIncomeDetails(BaseModel): recoveries: Optional[float]=0.0; parking_income: Optional[float]=0.0; laundry_income: Optional[float]=0.0; application_fees: Optional[float]=0.0; other_misc_income: Optional[float]=0.0
class IncomeSummary(BaseModel): gross_potential_rent: Optional[float]=0.0; loss_to_lease: Optional[float]=0.0; vacancy_credit_loss_details: VacancyCreditLossDetails; total_vacancy_credit_loss: Optional[float]=0.0; other_income_details: OtherIncomeDetails; total_other_income: Optional[float]=0.0; effective_gross_income: Optional[float]=0.0
class ExpenseDetails(BaseModel): property_taxes: Optional[float]=0.0; insurance: Optional[float]=0.0; property_management_fees: Optional[float]=0.0; repairs_maintenance: Optional[float]=0.0; utilities: Optional[float]=0.0; payroll: Optional[float]=0.0; admin_office_costs: Optional[float]=0.0; marketing_advertising: Optional[float]=0.0; other_opex: Optional[float]=0.0
class OperatingExpenses(BaseModel): expense_details: ExpenseDetails; total_operating_expenses: Optional[float]=0.0
class Reserves(BaseModel): replacement_reserves: Optional[float]=None; tenant_improvements: Optional[float]=None
class Financials(BaseModel): income_summary: IncomeSummary; operating_expenses: OperatingExpenses; net_operating_income: Optional[float]=0.0; reserves: Optional[Reserves]=None
class DetailedMergedResponse(BaseModel): document_type: str; period: str; financials: Financials; filename: Optional[str]=None; validation_warnings: Optional[List[str]]=None; error: Optional[str]=None
def validate_api_key(provided_key: Optional[str], request: Request) -> bool:
    """Validate API key from various sources."""
    # Get API key from environment within the function scope
    env_api_key = os.environ.get("API_KEY")
    
    if not env_api_key:
        logger.warning("No API_KEY set in environment, skipping authentication")
        return True
    
    # Try to get key from query params, headers, or Authorization header
    user_agent = request.headers.get("user-agent", "").lower() if request else ""
    param_api_key = provided_key
    auth_header = request.headers.get("Authorization") if request else None
    header_api_key = None  # Initialize this variable
    source = "Query Parameter"
    
    if auth_header and auth_header.startswith('Bearer '): 
        header_api_key = auth_header.replace('Bearer ', '')
        source = "Authorization Header"
    
    # Use header_api_key if available, otherwise use provided_key
    final_key = header_api_key if header_api_key else provided_key
    
    if final_key:
        if final_key == env_api_key: 
            logger.info(f"API key validated via {source}")
            return True
        else: 
            logger.warning(f"Invalid API key via {source}.")
            return False
    else: 
        logger.warning("No API key provided.")
        return False
@app.get("/health")
async def health_check(request: Request): 
    user_agent = request.headers.get("user-agent", "").lower();
    if "render" in user_agent or "health" in user_agent: 
        logger.info("Skipping auth for health check probe."); 
        return {"status": "healthy", "version": "2.2.4"} # Updated version
    param_api_key = request.query_params.get('api_key');
    if not validate_api_key(param_api_key, request): 
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"status": "healthy", "version": "2.2.4"} # Updated version
@app.post("/extract", response_model=DetailedMergedResponse)
async def extract_data(file: UploadFile=File(...), document_type: Optional[str]=Form(None), api_key: Optional[str]=Header(None, alias="x-api-key"), authorization: Optional[str]=Header(None), request: Request=None):
    auth_key = api_key;
    if not auth_key and authorization and authorization.startswith('Bearer '): auth_key = authorization.replace('Bearer ', '')
    if not validate_api_key(auth_key, request): raise HTTPException(status_code=401, detail="Unauthorized")
    start_process_time = time.time(); temp_file_path = None
    try:
        logger.info(f"Processing file: {file.filename} (type: {file.content_type})")
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or ".tmp")[1]) as temp_file: shutil.copyfileobj(file.file, temp_file); temp_file_path = temp_file.name; logger.info(f"Temp file: {temp_file_path}")
        preprocessed_data = preprocess_file(temp_file_path, file.content_type, file.filename)
        if isinstance(preprocessed_data, dict) and 'metadata' in preprocessed_data: preprocessed_data['metadata']['filename'] = file.filename
        if document_type: extraction_doc_type = map_noi_tool_to_extraction_type(document_type); period = classifier._extract_period_from_content(preprocessed_data) or classifier._extract_period_from_filename(file.filename)
        else: classification_result = classifier.classify(preprocessed_data); extraction_doc_type = classification_result['document_type']; period = classification_result['period']
        logger.info(f"Classified as '{extraction_doc_type}' for period '{period}'")
        extraction_result = extract_financial_data(preprocessed_data, extraction_doc_type, period)
        formatted_result = validate_and_format_data(extraction_result)
        logger.info(f"Processed {file.filename} in {time.time() - start_process_time:.2f}s"); return formatted_result
    except HTTPException as http_exc: raise http_exc
    except FileNotFoundError as fnf_err: logger.error(f"FNF error: {fnf_err}", exc_info=True); raise HTTPException(status_code=404, detail=f"Processing error: {fnf_err}")
    except ValueError as val_err: logger.error(f"Value error: {val_err}", exc_info=True); raise HTTPException(status_code=400, detail=f"Invalid input: {val_err}")
    except Exception as e: logger.error(f"Error processing {file.filename}: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Internal server error: {file.filename}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try: os.unlink(temp_file_path); logger.info(f"Deleted temp file: {temp_file_path}")
            except Exception as del_e: logger.error(f"Error deleting temp file {temp_file_path}: {del_e}")
@app.post("/extract-batch", response_model=List[DetailedMergedResponse])
async def extract_batch(files: List[UploadFile]=File(...), document_types: Optional[str]=Form(None), api_key: Optional[str]=Header(None, alias="x-api-key"), authorization: Optional[str]=Header(None), request: Request=None):
    auth_key = api_key;
    if not auth_key and authorization and authorization.startswith('Bearer '): auth_key = authorization.replace('Bearer ', '')
    if not validate_api_key(auth_key, request): raise HTTPException(status_code=401, detail="Unauthorized")
    doc_types_map = {};
    if document_types:
        try: doc_types_map = json.loads(document_types); assert isinstance(doc_types_map, dict); logger.info(f"Doc types map: {doc_types_map}")
        except Exception as e: logger.error(f"Invalid doc_types JSON: {e}"); raise HTTPException(status_code=400, detail=f"Invalid document_types format: {e}")
    results = []; batch_start_time = time.time()
    for file in files:
        file_start_time = time.time(); temp_file_path = None; file_result_data: Dict[str, Any] = {}; extraction_doc_type = 'Unknown'; period = 'Unknown'
        try:
            logger.info(f"Batch processing: {file.filename} (type: {file.content_type})"); doc_type_hint = doc_types_map.get(file.filename)
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or ".tmp")[1]) as temp_file: shutil.copyfileobj(file.file, temp_file); temp_file_path = temp_file.name
            preprocessed_data = preprocess_file(temp_file_path, file.content_type, file.filename)
            if isinstance(preprocessed_data, dict) and 'metadata' in preprocessed_data: preprocessed_data['metadata']['filename'] = file.filename
            if doc_type_hint: extraction_doc_type = map_noi_tool_to_extraction_type(doc_type_hint); period = classifier._extract_period_from_content(preprocessed_data) or classifier._extract_period_from_filename(file.filename)
            else: classification_result = classifier.classify(preprocessed_data); extraction_doc_type = classification_result['document_type']; period = classification_result['period']
            extraction_result = extract_financial_data(preprocessed_data, extraction_doc_type, period)
            formatted_result = validate_and_format_data(extraction_result)
            formatted_result['filename'] = file.filename; file_result_data = formatted_result; logger.info(f"Processed batch file {file.filename} in {time.time() - file_start_time:.2f}s")
        except Exception as e:
            logger.error(f"Error processing batch file {file.filename}: {e}", exc_info=True); error_data = extractor._empty_result(extraction_doc_type, period); formatter = ValidationFormatter(); error_output = formatter._format_output(error_data); error_output['filename'] = file.filename or "unknown_file"; error_output['error'] = f"Failed to process: {str(e)}"; file_result_data = error_output
        finally:
             if temp_file_path and os.path.exists(temp_file_path):
                  try: os.unlink(temp_file_path)
                  except Exception as del_e: logger.error(f"Error deleting temp file {temp_file_path}: {del_e}")
        results.append(file_result_data)
    logger.info(f"Batch processing completed in {time.time() - batch_start_time:.2f}s"); return results

#################################################
# Helper Functions
#################################################
def preprocess_file(file_path: str, content_type: str = None, filename: str = None) -> Dict[str, Any]:
    """Preprocess a file using FilePreprocessor."""
    preprocessor = FilePreprocessor()
    return preprocessor.preprocess(file_path, content_type, filename)

def extract_financial_data(preprocessed_data: Dict[str, Any], document_type: str, period: Optional[str] = None) -> Dict[str, Any]:
    """Extract financial data using GPTDataExtractor."""
    extractor = GPTDataExtractor()
    return extractor.extract_data(preprocessed_data, document_type, period)

def map_noi_tool_to_extraction_type(noi_tool_type: str) -> str:
    """Map NOI Tool document type to extraction document type."""
    mapping = {
        "current_month_actuals": "Actual Income Statement",
        "prior_month_actuals": "Actual Income Statement",
        "current_month_budget": "Budget",
        "prior_year_actuals": "Prior Year Actual"
    }
    return mapping.get(noi_tool_type, "Unknown")

#################################################
# Initialize global instances
#################################################
classifier = DocumentClassifier()
extractor = GPTDataExtractor()

#################################################
# Main entry point
#################################################
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
