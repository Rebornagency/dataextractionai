import os
import logging
import re
from typing import Dict, Any, List, Tuple, Optional, Union
from openai import OpenAI
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('document_classifier')

class DocumentClassifier:
    """
    Class for classifying financial documents and extracting time periods
    using GPT-4, enhanced to work with labeled document types
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the document classifier

        Args:
            api_key: OpenAI API key (optional, can be set via environment variable)
        """
        # Set API key if provided, otherwise use environment variable
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            logger.warning("OpenAI API key not set. Please set OPENAI_API_KEY environment variable or provide it during initialization.")

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        # Document types we can classify
        self.document_types = [
            "Actual Income Statement",
            "Budget",
            "Prior Year Actual",
            "Unknown"
        ]

        # Mapping from NOI Tool document types to extraction tool document types
        self.document_type_mapping = {
            "current_month_actuals": "Actual Income Statement",
            "prior_month_actuals": "Actual Income Statement", # Could be classified differently if needed
            "current_month_budget": "Budget",
            "prior_year_actuals": "Prior Year Actual"
        }

    def classify(self, text_or_data: Union[str, Dict[str, Any]], known_document_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Classify document type and extract time period, with option to use known document type

        Args:
            text_or_data: Preprocessed text from the document or data dictionary
            known_document_type: Known document type from labeled upload (optional)

        Returns:
            Dict containing document type and period
        """
        logger.info("Classifying document and extracting time period")

        # If known document type is provided, use it directly
        if known_document_type:
            logger.info(f"Using known document type: {known_document_type}")
            # Map the NOI Tool document type to extraction tool document type
            if known_document_type in self.document_type_mapping:
                doc_type = self.document_type_mapping[known_document_type]
                logger.info(f"Mapped known document type '{known_document_type}' to '{doc_type}'")

                # Extract period from the document content
                period = self._extract_period_from_content(text_or_data)

                return {
                    'document_type': doc_type,
                    'period': period,
                    'method': 'known_type'
                }
            else:
                logger.warning(f"Unknown document type mapping for '{known_document_type}', falling back to extraction")

        # Extract text from the input for period extraction
        extracted_text = self._extract_text_from_input(text_or_data)

        # Try to extract period from filename if it's in the data
        filename = None
        if isinstance(text_or_data, dict) and 'metadata' in text_or_data and 'filename' in text_or_data['metadata']:
            filename = text_or_data['metadata']['filename']
            logger.info(f"Found filename in metadata: {filename}")

        # Try to determine document type from filename
        doc_type_from_filename = self._determine_type_from_filename(filename) if filename else None
        if doc_type_from_filename:
            logger.info(f"Determined document type from filename: {doc_type_from_filename}")
            period = self._extract_period_from_content(extracted_text) or self._extract_period_from_filename(filename)
            return {
                'document_type': doc_type_from_filename,
                'period': period,
                'method': 'filename'
            }

        # First try rule-based classification for efficiency
        rule_based_result = self._rule_based_classification(extracted_text)

        # If rule-based classification is confident, return the result
        if rule_based_result.get('confidence', 0) > 0.7:
            logger.info(f"Rule-based classification successful: {rule_based_result}")
            return {
                'document_type': rule_based_result['document_type'],
                'period': rule_based_result['period'],
                'method': 'rule_based'
            }

        # Otherwise, use GPT for classification
        gpt_result = self._gpt_classification(extracted_text)
        logger.info(f"GPT classification result: {gpt_result}")

        return {
            'document_type': gpt_result['document_type'],
            'period': gpt_result['period'],
            'method': 'gpt'
        }

    def _extract_text_from_input(self, text_or_data: Union[str, Dict[str, Any]]) -> str:
        """
        Extract text content from input which could be string or dictionary

        Args:
            text_or_data: Input data which could be string or dictionary

        Returns:
            Extracted text content
        """
        # Handle both string and dictionary input
        if isinstance(text_or_data, dict):
            # Extract text from dictionary if it's a dictionary
            # logger.info("Input is a dictionary, extracting text content")

            # Try to extract text from common dictionary structures
            if 'combined_text' in text_or_data:
                return text_or_data['combined_text']
            elif 'text' in text_or_data:
                if isinstance(text_or_data['text'], str):
                    return text_or_data['text']
                elif isinstance(text_or_data['text'], list) and len(text_or_data['text']) > 0:
                    # Join text from multiple pages
                    page_contents = []
                    for page in text_or_data['text']:
                        if isinstance(page, dict) and 'content' in page and page['content']:
                            page_contents.append(page['content'])
                        elif isinstance(page, str): # Handle case where list contains strings
                            page_contents.append(page)
                    return "\n\n".join(page_contents)

            elif 'content' in text_or_data and isinstance(text_or_data['content'], str):
                return text_or_data['content']
            elif 'data' in text_or_data and isinstance(text_or_data['data'], str):
                return text_or_data['data']
            elif 'sheets' in text_or_data and isinstance(text_or_data['sheets'], list):
                 # Extract text from Excel sheets text representation
                 if 'text_representation' in text_or_data and isinstance(text_or_data['text_representation'], list):
                      return "\n\n".join(text_or_data['text_representation'])

            # If we couldn't extract text using known structures, convert the entire dictionary to a string
            logger.warning("Could not find specific text field in dictionary, using JSON string representation")
            try:
                # Limit the depth and length to avoid overly long strings
                return json.dumps(text_or_data, indent=2, default=str, ensure_ascii=False) # Use default=str for non-serializable types
            except Exception as e:
                logger.error(f"Error converting dict to JSON string: {e}")
                return str(text_or_data)[:2000] # Fallback to basic string conversion, limited length
        else:
            # Use the input directly if it's already a string
            if isinstance(text_or_data, str):
                return text_or_data
            else:
                # Convert to string if it's not a string
                logger.warning(f"Converted non-string input to string: {type(text_or_data)}")
                return str(text_or_data)

    def _extract_period_from_content(self, text_or_data: Union[str, Dict[str, Any]]) -> Optional[str]:
        """
        Extract period information from document content
        Uses more robust regex patterns.

        Args:
            text_or_data: Preprocessed text from the document or data dictionary

        Returns:
            Extracted period or None if not found
        """
        # Extract text from input
        text = self._extract_text_from_input(text_or_data)
        if not text:
            return None

        # Define month names and abbreviations for regex
        months_full = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        months_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        months_pattern = '|'.join(months_full + months_abbr)

        # Pattern 1: Month and Year (e.g., "For the Month Ended March 31, 2025", "Jan 2025", "January, 2025")
        # Allows optional day and separators like comma, space, underscore, hyphen
        month_year_pattern = rf'(?:For the (?:Month|Period) Ended\s+)?({months_pattern})[\s.,_-]*(\d{{1,2}}[\s.,_-]*)?(\d{{4}})'

        # Pattern 2: Quarter and Year (e.g., "Q1 2025", "First Quarter 2025", "Quarter Ended March 31, 2025")
        quarter_words = ['First', 'Second', 'Third', 'Fourth']
        quarter_pattern = rf'(?:(Q[1-4])|({"|".join(quarter_words)})\s+Quarter)[\s.,_-]*(\d{{4}})|(?:Quarter Ended\s+(?:{months_pattern})\s+\d{{1,2}}[\s.,_-]*)(\d{{4}})'

        # Pattern 3: Year only (e.g., "For the Year Ended December 31, 2025", "FY 2025", "Calendar Year 2025")
        year_pattern = r'(?:For the Year Ended|FY|Fiscal Year|Calendar Year)[\s.,_-]*(\d{4})'

        # Search for patterns in order of specificity
        # 1. Month and Year
        month_year_match = re.search(month_year_pattern, text, re.IGNORECASE)
        if month_year_match:
            month = month_year_match.group(1)
            year = month_year_match.group(3)
            # Standardize month name if needed (e.g., Jan -> January)
            for i, abbr in enumerate(months_abbr):
                if month.lower().startswith(abbr.lower()):
                    month = months_full[i]
                    break
            return f"{month} {year}"

        # 2. Quarter and Year
        quarter_match = re.search(quarter_pattern, text, re.IGNORECASE)
        if quarter_match:
            if quarter_match.group(1): # Q1-Q4 format
                return f"{quarter_match.group(1)} {quarter_match.group(3)}"
            elif quarter_match.group(2): # First-Fourth Quarter format
                 return f"{quarter_match.group(2)} Quarter {quarter_match.group(3)}"
            elif quarter_match.group(4): # Quarter Ended format
                 return f"Quarter Ended {quarter_match.group(4)}"

        # 3. Year only
        year_match = re.search(year_pattern, text, re.IGNORECASE)
        if year_match:
            return f"Year {year_match.group(1)}"

        # If no pattern matched, return None
        return None

    def _extract_period_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract period information from filename
        Uses regex patterns to find date information in the filename.

        Args:
            filename: Original filename

        Returns:
            Extracted period or None if not found
        """
        if not filename:
            return None

        # Define month names and abbreviations for regex
        months_full = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        months_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        months_pattern = '|'.join(months_full + months_abbr)
        
        # Pattern for numeric dates (MM_YYYY, MM-YYYY, etc.)
        numeric_date_pattern = r'(?:^|\D)(?:0?[1-9]|1[0-2])[\-_\.](?:20\d{2}|\d{2})(?:\D|$)'
        
        # Pattern for month name and year (Jan_2025, January-2025, etc.)
        month_year_pattern = rf'(?:{months_pattern})[\-_\.\s]*(?:20\d{{2}}|\d{{2}})(?:\D|$)'
        
        # Pattern for quarter (Q1_2025, Q2-2025, etc.)
        quarter_pattern = r'Q[1-4][\-_\.\s]*(?:20\d{2}|\d{2})(?:\D|$)'
        
        # Check patterns
        if re.search(month_year_pattern, filename, re.IGNORECASE):
            match = re.search(month_year_pattern, filename, re.IGNORECASE)
            text = match.group(0).strip()
            # Extract month and year
            for month in months_full + months_abbr:
                if month.lower() in text.lower():
                    # Find the month in the text
                    month_index = text.lower().find(month.lower())
                    month_text = text[month_index:month_index+len(month)]
                    # Find the year after the month
                    year_match = re.search(r'(20\d{2}|\d{2})', text[month_index+len(month):])
                    if year_match:
                        year_text = year_match.group(0)
                        # Ensure 2-digit years are converted to 4-digit
                        if len(year_text) == 2:
                            year_text = '20' + year_text
                        # Standardize month name if needed
                        for i, abbr in enumerate(months_abbr):
                            if month_text.lower().startswith(abbr.lower()):
                                month_text = months_full[i]
                                break
                        return f"{month_text} {year_text}"
        
        # Check for quarter pattern
        elif re.search(quarter_pattern, filename, re.IGNORECASE):
            match = re.search(quarter_pattern, filename, re.IGNORECASE)
            text = match.group(0).strip()
            # Extract quarter and year
            quarter_match = re.search(r'Q[1-4]', text, re.IGNORECASE)
            year_match = re.search(r'(20\d{2}|\d{2})', text)
            if quarter_match and year_match:
                quarter_text = quarter_match.group(0).upper()
                year_text = year_match.group(0)
                # Ensure 2-digit years are converted to 4-digit
                if len(year_text) == 2:
                    year_text = '20' + year_text
                return f"{quarter_text} {year_text}"
        
        # Check for numeric date pattern
        elif re.search(numeric_date_pattern, filename):
            match = re.search(numeric_date_pattern, filename)
            text = match.group(0).strip()
            # Extract month and year
            parts = re.split(r'[\-_\.]', text)
            # Filter out non-numeric parts and convert to integers
            numeric_parts = [int(part) for part in parts if part.isdigit()]
            if len(numeric_parts) >= 2:
                month_num = numeric_parts[0]
                year_num = numeric_parts[1]
                # Ensure month is valid
                if 1 <= month_num <= 12:
                    # Ensure year is 4 digits
                    if year_num < 100:
                        year_num = 2000 + year_num
                    # Convert month number to name
                    month_name = months_full[month_num - 1]
                    return f"{month_name} {year_num}"
        
        # If no pattern matched, return None
        return None

    def _determine_type_from_filename(self, filename: str) -> Optional[str]:
        """
        Try to determine document type from filename
        
        Args:
            filename: Original filename
            
        Returns:
            Document type or None if can't be determined
        """
        if not filename:
            return None
            
        # Convert to lowercase for case-insensitive matching
        filename_lower = filename.lower()
        
        # Check for budget indicators
        budget_indicators = ['budget', 'budgeted', 'forecast', 'projected', 'plan']
        if any(indicator in filename_lower for indicator in budget_indicators):
            return "Budget"
            
        # Check for actual indicators
        actual_indicators = ['actual', 'statement', 'income', 'p&l', 'profit', 'loss']
        if any(indicator in filename_lower for indicator in actual_indicators):
            # Check if it's prior year
            prior_indicators = ['prior', 'previous', 'last', 'py', 'ly']
            if any(indicator in filename_lower for indicator in prior_indicators):
                return "Prior Year Actual"
            else:
                return "Actual Income Statement"
                
        # If no clear indicators, return None
        return None

    def _rule_based_classification(self, text: str) -> Dict[str, Any]:
        """
        Classify document using rule-based approach
        
        Args:
            text: Preprocessed text from the document
            
        Returns:
            Dict with document type, period, and confidence
        """
        # Default result
        result = {
            'document_type': "Unknown",
            'period': None,
            'confidence': 0.0
        }
        
        if not text:
            return result
            
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Extract period first
        period = self._extract_period_from_content(text)
        result['period'] = period
        
        # Check for budget indicators
        budget_indicators = ['budget', 'budgeted', 'forecast', 'projected', 'plan']
        budget_score = sum(10 for indicator in budget_indicators if indicator in text_lower)
        
        # Check for actual indicators
        actual_indicators = ['actual', 'statement', 'income statement', 'p&l', 'profit and loss']
        actual_score = sum(10 for indicator in actual_indicators if indicator in text_lower)
        
        # Check for prior year indicators
        prior_indicators = ['prior year', 'previous year', 'last year', 'py', 'ly']
        prior_score = sum(5 for indicator in prior_indicators if indicator in text_lower)
        
        # Determine document type based on scores
        if budget_score > actual_score and budget_score > 20:
            result['document_type'] = "Budget"
            result['confidence'] = min(1.0, budget_score / 50)
        elif actual_score > 20:
            if prior_score > 15:
                result['document_type'] = "Prior Year Actual"
                result['confidence'] = min(1.0, (actual_score + prior_score) / 70)
            else:
                result['document_type'] = "Actual Income Statement"
                result['confidence'] = min(1.0, actual_score / 50)
                
        return result

    def _gpt_classification(self, text: str) -> Dict[str, Any]:
        """
        Classify document using GPT
        
        Args:
            text: Preprocessed text from the document
            
        Returns:
            Dict with document type and period
        """
        # Default result
        result = {
            'document_type': "Unknown",
            'period': None
        }
        
        if not text:
            return result
            
        # Truncate text if too long
        max_length = 8000  # Limit to avoid token limits
        if len(text) > max_length:
            # Take first and last parts
            first_part = text[:max_length//2]
            last_part = text[-max_length//2:]
            truncated_text = first_part + "\n...[content truncated]...\n" + last_part
        else:
            truncated_text = text
            
        try:
            # Create prompt for GPT
            prompt = f"""You are a financial document classifier. Analyze the following document text and determine:
1. The document type (Actual Income Statement, Budget, Prior Year Actual, or Unknown)
2. The time period the document covers (e.g., January 2025, Q1 2025, Year 2025)

Document text:
{truncated_text}

Respond in JSON format with the following structure:
{{
  "document_type": "one of: Actual Income Statement, Budget, Prior Year Actual, Unknown",
  "period": "the time period covered by the document or null if unknown"
}}
"""
            
            # Call GPT API
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial document classifier that responds only in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            gpt_result = json.loads(response_text)
            
            # Validate and extract results
            if 'document_type' in gpt_result:
                result['document_type'] = gpt_result['document_type']
                # Ensure document type is one of our valid types
                if result['document_type'] not in self.document_types:
                    result['document_type'] = "Unknown"
                    
            if 'period' in gpt_result:
                result['period'] = gpt_result['period']
                
        except Exception as e:
            logger.error(f"Error in GPT classification: {str(e)}")
            # If GPT fails, try to extract period using regex
            result['period'] = self._extract_period_from_content(text)
            
        return result
