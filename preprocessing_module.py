"""
Preprocessing Module for Real Estate NOI Analyzer
Extracts text and data from various file formats (PDF, Excel, CSV, TXT)
"""

import os
import io
import magic
import chardet
import pandas as pd
import pdfplumber
from typing import Dict, Any, List, Tuple, Optional
import logging
import json

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
    
    def preprocess(self, file_path: str) -> Dict[str, Any]:
        """
        Main method to preprocess a file
        
        Args:
            file_path: Path to the file to preprocess
            
        Returns:
            Dict containing extracted text/data and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file extension and check if supported
        _, ext = os.path.splitext(file_path)
        ext = ext.lower().lstrip('.')
        
        if ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Get file type using magic
        file_type = magic.from_file(file_path, mime=True)
        file_size = os.path.getsize(file_path)
        
        logger.info(f"Processing file: {file_path} ({file_type}, {file_size} bytes)")
        
        # Process the file based on its extension
        processor = self.supported_extensions[ext]
        extracted_content = processor(file_path)
        
        # Add metadata
        result = {
            'metadata': {
                'filename': os.path.basename(file_path),
                'file_type': file_type,
                'file_size': file_size,
                'extension': ext
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
                            df = pd.DataFrame(table)
                            # Use first row as header if it looks like a header
                            if self._is_header_row(df.iloc[0]):
                                df.columns = df.iloc[0]
                                df = df.iloc[1:]
                            
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
                # Read the sheet
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Store sheet data
                result['sheets'].append({
                    'name': sheet_name,
                    'data': df.to_dict(orient='records')
                })
                
                # Create text representation of the sheet
                text_rep = f"Sheet: {sheet_name}\n"
                text_rep += df.to_string(index=False)
                result['text_representation'].append(text_rep)
            
            # Combine all text representations
            result['combined_text'] = "\n\n".join(result['text_representation'])
            
        except Exception as e:
            logger.error(f"Error processing Excel file: {str(e)}")
            raise
        
        return result
    
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
            
            # Read CSV file
            df = pd.read_csv(file_path, encoding=encoding)
            
            # Store data
            result['data'] = df.to_dict(orient='records')
            
            # Create text representation
            result['text_representation'] = df.to_string(index=False)
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
            with open(file_path, 'r', encoding=encoding) as f:
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
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10000 bytes
        
        result = chardet.detect(raw_data)
        encoding = result['encoding'] or 'utf-8'
        
        return encoding
    
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
        
        # Replace multiple spaces with a single space
        text = ' '.join(text.split())
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive line breaks
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
        
        # Standardize number formats (e.g., 1,000.00 → 1000.00)
        # This is a simplified approach; a more robust solution would use regex
        
        return text
    
    def _is_header_row(self, row: pd.Series) -> bool:
        """
        Check if a row looks like a header row
        
        Args:
            row: DataFrame row to check
            
        Returns:
            True if the row looks like a header, False otherwise
        """
        # Convert all values to strings
        values = [str(v).lower() for v in row]
        
        # Check if any values contain typical header keywords
        header_keywords = ['total', 'income', 'expense', 'revenue', 'cost', 
                          'date', 'period', 'month', 'year', 'budget', 'actual']
        
        for value in values:
            for keyword in header_keywords:
                if keyword in value:
                    return True
        
        return False


def preprocess_file(file_path: str) -> Dict[str, Any]:
    """
    Convenience function to preprocess a file
    
    Args:
        file_path: Path to the file to preprocess
        
    Returns:
        Dict containing extracted text/data and metadata
    """
    preprocessor = FilePreprocessor()
    return preprocessor.preprocess(file_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python preprocessing_module.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        result = preprocess_file(file_path)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
