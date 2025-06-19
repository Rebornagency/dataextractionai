import os
import io
import logging
import json
import re
import magic
import chardet
import pandas as pd
import pdfplumber
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('preprocessing_module')

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
        Main method to preprocess a file

        Args:
            file_path: Path to the file to preprocess
            content_type: Content type of the file (optional)
            filename: Original filename (optional)

        Returns:
            Dict containing extracted text/data and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file extension from filename if provided, otherwise from file_path
        if filename and '.' in filename:
            _, ext = os.path.splitext(filename)
        else:
            _, ext = os.path.splitext(file_path)

        ext = ext.lower().lstrip('.')

        # Determine file type from content_type if provided
        detected_type = None
        if content_type:
            # Map content types to extensions
            content_type_map = {
                'application/pdf': 'pdf',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
                'application/vnd.ms-excel': 'xls',
                'text/csv': 'csv',
                'text/plain': 'txt'
            }

            # Check if content_type is in our map
            for ct, extension in content_type_map.items():
                if ct in content_type:
                    detected_type = extension
                    break

        # Use detected type if available, otherwise use extension
        if detected_type:
            ext = detected_type
            logger.info(f"Using file type from content_type: {ext}")

        # Check if file type is supported
        if ext not in self.supported_extensions:
            # Fallback to Excel processor for spreadsheet content types
            if content_type and ('spreadsheet' in content_type or 'excel' in content_type.lower()):
                logger.info(f"Unsupported extension {ext} but content type {content_type} indicates Excel, using Excel processor")
                processor = self._process_excel
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        else:
            # Process the file based on its extension
            processor = self.supported_extensions[ext]

        # Get file type using magic
        file_type = magic.from_file(file_path, mime=True)
        file_size = os.path.getsize(file_path)

        logger.info(f"Processing file: {file_path} ({file_type}, {file_size} bytes)")

        # Extract content
        extracted_content = processor(file_path)

        # Add metadata
        result = {
            'metadata': {
                'filename': filename or os.path.basename(file_path),
                'file_type': file_type,
                'file_size': file_size,
                'extension': ext,
                'content_type': content_type
            },
            'content': extracted_content
        }

        return result

    def _process_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Process PDF files using pdfplumber

        Args:
            file_path: Path to the PDF file

        Returns:
            Dict containing extracted text and tables
        """
        logger.info(f"Extracting content from PDF: {file_path}")
        result = {
            'text': [],
            'tables': []
        }

        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        result['text'].append({
                            'page': i + 1,
                            'content': self._clean_text(page_text)
                        })

                    # Extract tables
                    tables = page.extract_tables()
                    for j, table in enumerate(tables):
                        if table:
                            # Convert table to DataFrame for easier processing
                            # Handle potential None values before creating DataFrame
                            cleaned_table = [[str(cell) if cell is not None else '' for cell in row] for row in table]
                            df = pd.DataFrame(cleaned_table)
                            # Use first row as header if it looks like a header
                            if not df.empty and self._is_header_row(df.iloc[0]):
                                df.columns = df.iloc[0]
                                df = df[1:].reset_index(drop=True) # Reset index after removing header

                            result['tables'].append({
                                'page': i + 1,
                                'table_index': j,
                                'data': df.to_dict(orient='records')
                            })
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

        # Combine all text for easier processing
        all_text = "\n\n".join([page['content'] for page in result['text']])
        result['combined_text'] = all_text

        return result

    def _process_excel(self, file_path: str) -> Dict[str, Any]:
        """
        Process Excel files using pandas and openpyxl

        Args:
            file_path: Path to the Excel file

        Returns:
            Dict containing extracted sheets and data
        """
        logger.info(f"Extracting content from Excel: {file_path}")
        result = {
            'sheets': [],
            'text_representation': []
        }

        try:
            # Get list of sheet names
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names

            for sheet_name in sheet_names:
                # Read the sheet, handle potential parsing issues
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None) # Read without assuming header initially
                    # Attempt to find a meaningful header row
                    header_row_index = self._find_header_row(df)
                    if header_row_index is not None:
                         # Reread with the correct header row
                         df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row_index)
                    else:
                         # If no clear header, use default or generate simple headers
                         df = pd.read_excel(file_path, sheet_name=sheet_name) # Let pandas guess

                    # Convert NaN to None for JSON compatibility
                    df = df.replace({np.nan: None})

                    # Store sheet data
                    result['sheets'].append({
                        'name': sheet_name,
                        'data': df.to_dict(orient='records')
                    })

                    # Create text representation of the sheet
                    text_rep = f"Sheet: {sheet_name}\n"
                    text_rep += df.to_string(index=False, na_rep='None') # Represent None as 'None' string
                    result['text_representation'].append(text_rep)
                except Exception as sheet_error:
                     logger.warning(f"Could not process sheet '{sheet_name}' in {file_path}: {sheet_error}")


            # Combine all text representations
            result['combined_text'] = "\n\n".join(result['text_representation'])

        except Exception as e:
            logger.error(f"Error processing Excel file: {str(e)}")
            raise

        return result

    def _find_header_row(self, df: pd.DataFrame, max_rows_to_check=10) -> Optional[int]:
        """ Tries to find the most likely header row in the first few rows """
        if df.empty:
            return None
        for i in range(min(max_rows_to_check, len(df))):
            row = df.iloc[i]
            # Simple heuristic: check if row contains common header keywords and few numbers
            if self._is_header_row(row):
                return i
        return None # Return None if no clear header found

    def _process_csv(self, file_path: str) -> Dict[str, Any]:
        """
        Process CSV files using pandas

        Args:
            file_path: Path to the CSV file

        Returns:
            Dict containing extracted data
        """
        logger.info(f"Extracting content from CSV: {file_path}")
        result = {}

        try:
            # Detect encoding
            encoding = self._detect_encoding(file_path)

            # Read CSV file, handle potential parsing issues
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False) # low_memory=False can help with mixed types
            # Convert NaN to None for JSON compatibility
            df = df.replace({np.nan: None})

            # Store data
            result['data'] = df.to_dict(orient='records')

            # Create text representation
            result['text_representation'] = df.to_string(index=False, na_rep='None')
            result['combined_text'] = result['text_representation']

        except Exception as e:
            logger.error(f"Error processing CSV file: {str(e)}")
            raise

        return result

    def _process_txt(self, file_path: str) -> Dict[str, Any]:
        """
        Process TXT files

        Args:
            file_path: Path to the TXT file

        Returns:
            Dict containing extracted text
        """
        logger.info(f"Extracting content from TXT: {file_path}")
        result = {}

        try:
            # Detect encoding
            encoding = self._detect_encoding(file_path)

            # Read text file
            with open(file_path, 'r', encoding=encoding, errors='replace') as f: # Replace errors
                text = f.read()

            # Clean text
            cleaned_text = self._clean_text(text)

            # Store data
            result['text'] = cleaned_text
            result['combined_text'] = cleaned_text

        except Exception as e:
            logger.error(f"Error processing TXT file: {str(e)}")
            raise

        return result

    def _detect_encoding(self, file_path: str) -> str:
        """
        Detect file encoding

        Args:
            file_path: Path to the file

        Returns:
            Detected encoding
        """
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10000 bytes

            result = chardet.detect(raw_data)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'
            # Handle specific cases like 'ascii' -> 'utf-8' for broader compatibility
            if encoding.lower() == 'ascii':
                encoding = 'utf-8'
            logger.info(f"Detected encoding: {encoding} with confidence {result['confidence']}")
            return encoding
        except Exception as e:
             logger.warning(f"Could not detect encoding for {file_path}, defaulting to utf-8. Error: {e}")
             return 'utf-8'


    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Replace multiple spaces/tabs with a single space
        text = re.sub(r'\s+', ' ', text)

        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove excessive line breaks
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')

        # Remove leading/trailing whitespace from each line
        text = "\n".join([line.strip() for line in text.split('\n')])

        return text.strip() # Remove leading/trailing whitespace from the whole text

    def _is_header_row(self, row: pd.Series) -> bool:
        """
        Check if a row looks like a header row (improved heuristic)

        Args:
            row: DataFrame row to check

        Returns:
            True if the row looks like a header, False otherwise
        """
        if row.isnull().all(): # Skip entirely null rows
             return False

        num_numeric = 0
        num_non_numeric = 0
        header_keywords = ['total', 'income', 'expense', 'revenue', 'cost',
                          'date', 'period', 'month', 'year', 'budget', 'actual',
                          'rent', 'fee', 'tax', 'insurance', 'utility', 'maintenance',
                          'admin', 'payroll', 'marketing', 'vacancy', 'concession',
                          'debt', 'recover', 'parking', 'laundry', 'other', 'net',
                          'operating', 'gross', 'potential', 'effective', 'reserve',
                          'improvement', 'allowance', 'item', 'description', 'amount', 'fee'] # Added fee
        contains_keyword = False

        for v in row:
            if pd.isna(v) or str(v).strip() == '':
                continue
            v_str = str(v).lower().strip()
            try:
                # Attempt to convert to float, consider currency symbols/commas
                cleaned_v = v_str.replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
                if cleaned_v.strip().startswith('-'): # Handle trailing minus sign if needed
                    if cleaned_v.strip().endswith('-'):
                         cleaned_v = '-' + cleaned_v.strip().strip('-')
                float(cleaned_v)
                num_numeric += 1
            except ValueError:
                num_non_numeric += 1
                # Check for keywords only in non-numeric cells
                if any(keyword in v_str for keyword in header_keywords):
                    contains_keyword = True

        # Heuristic: More non-numeric fields than numeric, and contains at least one keyword
        # Or, mostly non-numeric and contains keywords
        total_valid_cells = num_numeric + num_non_numeric
        if total_valid_cells == 0:
             return False

        is_likely_header = (num_non_numeric > num_numeric or num_non_numeric / total_valid_cells > 0.6) and contains_keyword
        return is_likely_header

def preprocess_file(file_path: str, content_type: str = None, filename: str = None) -> Dict[str, Any]:
    """
    Convenience function to preprocess a file

    Args:
        file_path: Path to the file to preprocess
        content_type: Content type of the file (optional)
        filename: Original filename (optional)

    Returns:
        Dict containing extracted text/data and metadata
    """
    preprocessor = FilePreprocessor()
    return preprocessor.preprocess(file_path, content_type, filename)
