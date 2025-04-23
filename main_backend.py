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
