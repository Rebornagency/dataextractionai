"""
Data Extraction AI - Combined Module (v2.2.4 - All Fixes)
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
- Improved robustness in preprocessing and text extraction fallbacks

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
# Configure logging (Set level to DEBUG)
#################################################
logging.basicConfig(
    level=logging.DEBUG, # <-- Set to DEBUG to show detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Use a distinct logger name
logger = logging.getLogger('data_extraction_api_service_debug')

#################################################
# Preprocessing Module (Includes DEBUG LOGGING & Robustness)
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
            extracted_content = processor(file_path) # Call the specific processor
            # Ensure combined_text exists, even if empty after processing
            if 'combined_text' not in extracted_content:
                 logger.warning(f"Processor {processor.__name__} did not return 'combined_text'. Initializing empty.")
                 extracted_content['combined_text'] = ""
            elif not isinstance(extracted_content.get('combined_text'), str):
                 logger.warning(f"Processor {processor.__name__} returned non-string 'combined_text'. Converting.")
                 extracted_content['combined_text'] = str(extracted_content['combined_text'] or "")

        except Exception as proc_err:
            logger.error(f"Error during processing with {processor.__name__}: {proc_err}", exc_info=True)
            # Still create a basic structure with empty text on error
            extracted_content = {'combined_text': "", 'error': str(proc_err)}
            # Optionally re-raise if you want the API call to fail completely on preprocessing error
            # raise proc_err

        result = {
            'metadata': {
                'filename': filename or os.path.basename(file_path),
                'file_type_source': file_type_magic,
                'file_size_bytes': file_size,
                'detected_extension': final_ext,
                'original_content_type': content_type
            },
            'content': extracted_content # Contains 'combined_text' and potentially other structures
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

                    # Table extraction
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
            with open(file_path, 'rb') as f: raw_data = f.read(20000)
            result = chardet.detect(raw_data); encoding = result['encoding'] if result['encoding'] else 'utf-8'
            if encoding.lower() == 'ascii': encoding = 'utf-8'
            confidence = result['confidence'] if result else 0
            logger.info(f"Detected encoding: {encoding} with confidence {confidence:.2f}")
            if confidence is None or confidence < 0.7: logger.warning(f"Low confidence ({confidence}) for encoding '{encoding}'. Using fallback.")
            return encoding
        except Exception as e: logger.warning(f"Encoding detection failed: {e}. Defaulting to utf-8."); return 'utf-8'

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text extracted from documents."""
        if not text: return ""
        text = re.sub(r'\s+', ' ', text); text = text.strip()
        return text

    def _score_header_row(self, row: pd.Series) -> int:
        """ Scores a row based on likelihood of being a header. """
        if row.isnull().all(): return -1
        score = 0; num_numeric = 0; num_non_numeric = 0; contains_keyword = False; all_caps_count = 0
        header_keywords = ['total', 'income', 'expense', 'revenue', 'cost', 'date', 'period', 'month', 'year', 'budget', 'actual', 'rent', 'fee', 'tax', 'insurance', 'utility', 'maintenance', 'admin', 'payroll', 'marketing', 'vacancy', 'concession', 'debt', 'recover', 'parking', 'laundry', 'other', 'net', 'operating', 'gross', 'potential', 'effective', 'reserve', 'improvement', 'allowance', 'item', 'description', 'amount', 'gl', 'account']
        for v in row:
            if pd.isna(v) or str(v).strip() == '': continue
            v_str = str(v).strip(); v_lower = v_str.lower()
            try:
                cleaned_v = v_lower.replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
                if cleaned_v.strip().startswith('-'):
                    if cleaned_v.strip().endswith('-'): cleaned_v = '-' + cleaned_v.strip().strip('-')
                float(cleaned_v); num_numeric += 1
            except ValueError:
                num_non_numeric += 1
                if any(keyword in v_lower for keyword in header_keywords): contains_keyword = True
                if v_str.isupper() and len(v_str) > 1: all_caps_count += 1
        total_valid_cells = num_numeric + num_non_numeric
        if total_valid_cells == 0: return -1
        if contains_keyword: score += 2
        if num_non_numeric > num_numeric: score += 1
        if num_non_numeric / total_valid_cells > 0.7: score += 1
        if num_numeric / total_valid_cells < 0.2: score += 1
        if all_caps_count > total_valid_cells * 0.5: score += 1
        return score

    def _find_header_row(self, df: pd.DataFrame, max_rows_to_check=10) -> Optional[int]:
        """ Tries to find the most likely header row in the first few rows """
        if df.empty: return None
        best_score = -1; best_index = None
        for i in range(min(max_rows_to_check, len(df))):
            score = self._score_header_row(df.iloc[i])
            if score > best_score: best_score = score; best_index = i
        return best_index if best_score >= 2 else None

    def _is_header_row(self, row: pd.Series) -> bool:
        """Check if a row looks like a header row (calls scoring method)."""
        return self._score_header_row(row) >= 2 # Use score threshold


def preprocess_file(file_path: str, content_type: str = None, filename: str = None) -> Dict[str, Any]:
    """Convenience function to preprocess a file."""
    preprocessor = FilePreprocessor()
    return preprocessor.preprocess(file_path, content_type, filename)

#################################################
# Document Classifier Module (SyntaxError Fixed)
#################################################
class DocumentClassifier:
    """
    Class for classifying financial documents and extracting time periods
    using GPT-4, enhanced to work with labeled document types.
    Includes fix for SyntaxError.
    """
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the document classifier."""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key: logger.warning("OpenAI API key not set for DocumentClassifier."); self.client = None
        else: self.client = OpenAI(api_key=self.api_key)
        self.document_types = ["Actual Income Statement", "Budget", "Prior Year Actual", "Unknown"]
        self.document_type_mapping = {"current_month_actuals": "Actual Income Statement", "prior_month_actuals": "Actual Income Statement", "current_month_budget": "Budget", "prior_year_actuals": "Prior Year Actual"}

    def classify(self, text_or_data: Union[str, Dict[str, Any]], known_document_type: Optional[str] = None) -> Dict[str, Any]:
        """Classify document type and extract time period."""
        logger.info("Classifying document and extracting time period...")
        start_time = time.time()
        if known_document_type:
            mapped_type = self.document_type_mapping.get(known_document_type)
            if mapped_type:
                logger.info(f"Using known document type: '{known_document_type}' mapped to '{mapped_type}'")
                period = self._extract_period_from_content(text_or_data)
                return {'document_type': mapped_type, 'period': period, 'method': 'known_type', 'duration_ms': int((time.time() - start_time) * 1000)}
            else: logger.warning(f"Unknown mapping for known type '{known_document_type}', proceeding.")
        extracted_text = self._extract_text_from_input(text_or_data)
        filename = text_or_data.get('metadata', {}).get('filename') if isinstance(text_or_data, dict) else None
        if filename:
            doc_type_from_filename = self._determine_type_from_filename(filename)
            if doc_type_from_filename:
                logger.info(f"Determined type from filename '{filename}': {doc_type_from_filename}")
                period = self._extract_period_from_content(extracted_text) or self._extract_period_from_filename(filename)
                return {'document_type': doc_type_from_filename, 'period': period, 'method': 'filename', 'duration_ms': int((time.time() - start_time) * 1000)}
        rule_based_result = self._rule_based_classification(extracted_text)
        if rule_based_result.get('confidence', 0) > 0.75:
            logger.info(f"Rule-based classification successful: {rule_based_result}")
            return {**rule_based_result, 'method': 'rule_based', 'duration_ms': int((time.time() - start_time) * 1000)}
        if not self.client: logger.error("OpenAI client not available for GPT fallback."); return {**rule_based_result, 'method': 'rule_based_low_confidence', 'duration_ms': int((time.time() - start_time) * 1000)}
        logger.info("Falling back to GPT for classification.")
        gpt_result = self._gpt_classification(extracted_text)
        logger.info(f"GPT classification result: {gpt_result}")
        return {**gpt_result, 'method': 'gpt', 'duration_ms': int((time.time() - start_time) * 1000)}

    def _extract_text_from_input(self, text_or_data: Union[str, Dict[str, Any]]) -> str:
        """Extract text content from input (string or dictionary) with better fallbacks."""
        logger.debug(f"DEBUG: _extract_text_from_input called with type: {type(text_or_data)}")
        if isinstance(text_or_data, dict):
            # Access content within the 'content' key from preprocessor output
            content_data = text_or_data.get('content', {})
            if not isinstance(content_data, dict): # Handle case where content isn't a dict
                logger.warning("Content field is not a dictionary, using string representation.")
                return str(content_data)[:5000]

            combined_text = content_data.get('combined_text')
            if combined_text and isinstance(combined_text, str) and combined_text.strip():
                logger.debug("DEBUG: Using 'combined_text'.")
                return combined_text
            text_list = content_data.get('text')
            if text_list and isinstance(text_list, list):
                page_contents = [p.get('content', '') if isinstance(p, dict) else str(p) for p in text_list]
                joined_text = "\n\n".join(filter(None, page_contents))
                if joined_text.strip():
                    logger.debug("DEBUG: Using joined 'text' list.")
                    return joined_text
            text_repr_list = content_data.get('text_representation')
            if text_repr_list and isinstance(text_repr_list, list):
                 joined_repr = "\n\n".join(filter(None, text_repr_list))
                 if joined_repr.strip():
                      logger.debug("DEBUG: Using joined 'text_repr' list.")
                      return joined_repr
            logger.warning("DEBUG: Could not find standard text fields in content, using JSON fallback of content dict.")
            try:
                # Dump only the content dictionary, not the whole input
                return json.dumps(content_data, default=str, ensure_ascii=False, indent=None, separators=(',', ':'))[:5000]
            except Exception as json_err:
                 logger.error(f"JSON dump fallback error: {json_err}")
                 return str(content_data)[:5000]
        elif isinstance(text_or_data, str):
            logger.debug("DEBUG: Input is string.")
            return text_or_data
        else:
            logger.warning(f"Input not dict/str: {type(text_or_data)}")
            return str(text_or_data)

    def _extract_period_from_content(self, text_or_data: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extract period information from document content using robust regex."""
        text = self._extract_text_from_input(text_or_data)
        if not text or len(text) < 4: return None
        months_full=['January','February','March','April','May','June','July','August','September','October','November','December']
        months_abbr=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        months_pattern='|'.join(months_full+months_abbr); year_pattern=r'(20\d{2})'
        pattern1 = rf'(?:For the (?:Month|Period)\s+(?:Ended|Ending)\s+)?({months_pattern})\s+(\d{{1,2}})[\s,]+{year_pattern}'
        match1 = re.search(pattern1, text, re.IGNORECASE)
        if match1:
            month_full=self._standardize_month(match1.group(1),months_full,months_abbr)
            logger.debug(f"Period pattern 1: {month_full} {match1.group(3)}")
            return f"{month_full} {match1.group(3)}"
        pattern2 = rf'({months_pattern})[\s.,_-]+{year_pattern}'
        match2 = re.search(pattern2, text, re.IGNORECASE)
        if match2:
            month_full=self._standardize_month(match2.group(1),months_full,months_abbr)
            logger.debug(f"Period pattern 2: {month_full} {match2.group(2)}")
            return f"{month_full} {match2.group(2)}"
        quarter_words=['First','Second','Third','Fourth']
        pattern3 = rf'(?:(Q[1-4])|({"|".join(quarter_words)})\s+Quarter)[\s.,_-]*{year_pattern}|(?:Quarter\s+(?:Ended|Ending)\s+(?:{months_pattern})\s+\d{{1,2}}[\s.,_-]*)({year_pattern})'
        match3 = re.search(pattern3, text, re.IGNORECASE)
        if match3:
            year_str=match3.group(3) or match3.group(4)
            if match3.group(1): period_str=f"{match3.group(1)} {year_str}"
            elif match3.group(2): period_str=f"{match3.group(2)} Quarter {year_str}"
            else: period_str=f"Quarter in {year_str}"
            logger.debug(f"Period pattern 3: {period_str}")
            return period_str
        pattern4 = rf'(?:For the Year\s+(?:Ended|Ending)|FY|Fiscal Year|Calendar Year)[\s.,_-]*({year_pattern})'
        match4 = re.search(pattern4, text, re.IGNORECASE)
        if match4:
            period_str=f"Year {match4.group(1)}"
            logger.debug(f"Period pattern 4: {period_str}")
            return period_str
        for match5 in re.finditer(rf'\b({year_pattern})\b', text):
            year_str=match5.group(1)
            context_window=text[max(0,match5.start()-40):min(len(text),match5.end()+40)]
            if any(kw in context_window.lower() for kw in ['period','month','year','date','quarter','budget','actual','statement','report','ended','ending']):
                 month_nearby_match=re.search(rf'({months_pattern})',context_window,re.IGNORECASE)
                 if month_nearby_match:
                      month_full=self._standardize_month(month_nearby_match.group(1),months_full,months_abbr)
                      period_str=f"{month_full} {year_str}"
                      logger.debug(f"Period pattern 5 (Y+M Context): {period_str}")
                      return period_str
                 else:
                      period_str=f"Year {year_str}"
                      logger.debug(f"Period pattern 5 (Y Context): {period_str}")
                      return period_str
        logger.warning("Could not extract period from content.")
        return None

    def _standardize_month(self, month_str: str, months_full: List[str], months_abbr: List[str]) -> str:
        """Converts abbreviated month to full month name."""
        month_lower = month_str.lower()
        for i, abbr in enumerate(months_abbr):
            if month_lower.startswith(abbr.lower()):
                return months_full[i]
        return month_str

    def _determine_type_from_filename(self, filename: Optional[str]) -> Optional[str]:
        """Determine document type from filename keywords."""
        # --- THIS FUNCTION IS FIXED (Correct Indentation/Structure) ---
        if not filename:
            return None
        filename_lower = filename.lower()
        is_budget = 'budget' in filename_lower or 'bdgt' in filename_lower
        is_actual = 'actual' in filename_lower or 'is' in filename_lower or 'p&l' in filename_lower or 'income statement' in filename_lower or 'profit loss' in filename_lower
        is_prior_year = 'prior year' in filename_lower or 'py' in filename_lower or 'previous year' in filename_lower or 'last year' in filename_lower

        if is_budget:
            return "Budget"
        if is_actual:
            if is_prior_year:
                return "Prior Year Actual"
            else:
                return "Actual Income Statement"
        return None # Return None if neither matched
        # --- END OF FIX ---

    def _extract_period_from_filename(self, filename: str) -> Optional[str]:
        """Extract period information from filename (e.g., "IS_Jan_2025.pdf")."""
        if not filename: return None
        name_part = os.path.splitext(filename)[0]; name_part = re.sub(r'[_\-\.]+', ' ', name_part)
        months_full = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        months_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        months_pattern = '|'.join(months_full + months_abbr); year_pattern = r'(20\d{2})'
        month_year_match = re.search(rf'({months_pattern})\s+{year_pattern}', name_part, re.IGNORECASE)
        if month_year_match: return f"{self._standardize_month(month_year_match.group(1), months_full, months_abbr)} {month_year_match.group(2)}"
        year_month_match = re.search(rf'{year_pattern}\s+({months_pattern})', name_part, re.IGNORECASE)
        if year_month_match: return f"{self._standardize_month(year_month_match.group(2), months_full, months_abbr)} {year_month_match.group(1)}"
        quarter_match = re.search(rf'(Q[1-4])\s*{year_pattern}', name_part, re.IGNORECASE)
        if quarter_match: return f"{quarter_match.group(1)} {quarter_match.group(2)}"
        quarter_match_rev = re.search(rf'{year_pattern}\s*(Q[1-4])', name_part, re.IGNORECASE)
        if quarter_match_rev: return f"{quarter_match_rev.group(2)} {quarter_match_rev.group(1)}"
        year_match = re.search(year_pattern, name_part)
        if year_match and any(kw in name_part.lower() for kw in ['budget', 'actual', 'year', 'report']): return f"Year {year_match.group(1)}"
        return None

    def _rule_based_classification(self, text: str) -> Dict[str, Any]:
        """Attempt rule-based classification."""
        # --- THIS FUNCTION IS REFORMATTED FOR READABILITY ---
        result = {'document_type': 'Unknown', 'period': None, 'confidence': 0.0}
        if not text: return result

        text_lower = text.lower()
        doc_type = 'Unknown'
        type_confidence = 0.0

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
                type_confidence = min(0.7 + actual_score * 0.05, 0.9)

        period = self._extract_period_from_content(text)
        current_confidence = type_confidence
        if period:
            result['period'] = period
            current_confidence = min(current_confidence + 0.15, 1.0)

        result['document_type'] = doc_type
        result['confidence'] = current_confidence
        return result
        # --- END OF REFORMAT ---

    def _gpt_classification(self, text: str) -> Dict[str, Any]:
        """Classify document using GPT-4."""
        # --- THIS FUNCTION IS REFORMATTED FOR READABILITY ---
        default_result = {'document_type': 'Unknown', 'period': None}
        if not self.client: logger.error("OpenAI client not initialized."); return default_result
        text_sample = text[:3000] if text else ""
        if not text_sample: logger.warning("Empty text for GPT classification."); return default_result

        prompt = f"""Analyze the beginning of this financial document to determine its type and the period it covers.
Document Types: {self.document_types}
Period Format: Extract the most specific period available (e.g., "Month Year", "Quarter Year", "Year").
Document Sample:
---
{text_sample}
---
Respond ONLY with a JSON object containing 'document_type' and 'period' fields. Example: {{"document_type": "Actual Income Statement", "period": "March 2025"}}"""

        try:
            response = self.client.chat.completions.create(model="gpt-4", messages=[{"role": "system", "content": "You are an expert financial document classifier... Respond only JSON."}, {"role": "user", "content": prompt}], temperature=0.1, max_tokens=100)
            response_text = response.choices[0].message.content.strip(); logger.info(f"GPT classification raw: {response_text}")
            try:
                cleaned_response_text = response_text.strip('`').strip()
                if cleaned_response_text.startswith('json'): cleaned_response_text = cleaned_response_text[4:].strip()
                result = json.loads(cleaned_response_text)
                if 'document_type' in result and 'period' in result:
                    if result['document_type'] not in self.document_types: result['document_type'] = 'Unknown'
                    logger.info(f"Parsed GPT classification: {result}"); return result
                else: raise json.JSONDecodeError("Missing fields", cleaned_response_text, 0)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON from GPT classification: {e}")
                doc_type_match = re.search(r'"document_type":\s*"([^"]+)"', response_text); period_match = re.search(r'"period":\s*"([^"]*)"', response_text)
                doc_type_res = doc_type_match.group(1) if doc_type_match else 'Unknown'; period_res = period_match.group(1) if period_match else None
                if doc_type_res not in self.document_types: doc_type_res = 'Unknown'
                logger.warning(f"Falling back to regex for classification: Type='{doc_type_res}', Period='{period_res}'"); return {'document_type': doc_type_res, 'period': period_res}
        except (RateLimitError, APIError, APITimeoutError) as api_err: logger.error(f"OpenAI API error: {api_err}"); return default_result
        except Exception as e: logger.error(f"Unexpected error in GPT classification: {e}", exc_info=True); return default_result
        # --- END OF REFORMAT ---

# Create an instance of the classifier for direct import
classifier = DocumentClassifier()

# Convenience functions
def classify_document(text_or_data: Union[str, Dict[str, Any]], known_document_type: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Classify document type and extract time period using the classifier instance."""
    result = classifier.classify(text_or_data, known_document_type)
    return result['document_type'], result['period']

def map_noi_tool_to_extraction_type(noi_tool_type: str) -> str:
    """Map NOI Tool document type hint to the extraction tool's expected type."""
    return classifier.document_type_mapping.get(noi_tool_type, "Unknown")


#################################################
# GPT Data Extractor Module (SyntaxError Fixed in _extract_with_gpt)
#################################################
class GPTDataExtractor:
    """Includes fix for NumPy 2.0 float type error and missing except block."""
    def __init__(self, api_key: Optional[str] = None, sample_limit: int = 4000):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY"); self.sample_limit = sample_limit
        if not self.api_key: logger.warning("OpenAI API key not set."); self.client = None
        else: self.client = OpenAI(api_key=self.api_key)

    def extract(self, text_or_data: Union[str, Dict[str, Any]], document_type: str, period: Optional[str] = None) -> Dict[str, Any]:
        """Extracts detailed financial data."""
        if not self.client: logger.error("OpenAI client not initialized."); return self._empty_result(document_type, period)
        logger.info(f"Extracting data from {document_type} for {period or 'Unknown'}"); text = classifier._extract_text_from_input(text_or_data)
        if not text: logger.warning("No text for extraction."); return self._empty_result(document_type, period)
        prompt = self._create_extraction_prompt(text, document_type, period); extraction_result = self._extract_with_gpt(prompt)
        if not extraction_result or not isinstance(extraction_result, dict): logger.warning("Invalid GPT result."); extraction_result = self._empty_result(document_type, period)
        else:
             extraction_result['document_type'] = extraction_result.get('document_type', document_type); extraction_result['period'] = extraction_result.get('period', period or 'Unknown'); default_empty = self._empty_result()
             for key in default_empty.keys():
                 if key not in ['document_type', 'period'] and key not in extraction_result: extraction_result[key] = default_empty[key]
        self._validate_and_clean_extraction_result(extraction_result); return extraction_result

    def _create_extraction_prompt(self, text: str, document_type: str, period: Optional[str] = None) -> str:
        """Creates the detailed prompt for GPT-4 extraction, including Application Fees."""
        text_sample = text[:self.sample_limit]; period_context = f" for period '{period}'" if period else ""
        # Using the full prompt structure from previous steps
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
            * `application_fees`: Fees collected from rental applications.
            * `other_misc_income`: Other income (storage, signage, late fees, etc.). Sum if multiple lines exist.
            * `total_other_income`: Sum of ALL above other income items (calculate if not explicitly stated).

        4.  **EFFECTIVE GROSS INCOME (EGI):**
            * `effective_gross_income`: Calculate as (Gross Potential Rent - Total Vacancy & Credit Loss + Total Other Income). Verify if stated explicitly.

        5.  **OPERATING EXPENSES (OpEx):**
            * `property_taxes`: Real estate taxes.
            * `insurance`: Property insurance costs.
            * `property_management_fees`: Fees paid for management.
            * `repairs_maintenance`: General repairs 
