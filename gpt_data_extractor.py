import os
import logging
from typing import Dict, Any, List, Optional, Union
from openai import OpenAI
import json
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gpt_data_extractor')

class GPTDataExtractor:
    """
    Class for extracting financial data from documents using GPT-4
    Enhanced to extract detailed income components and application fees
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the data extractor

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

    def extract_data(self, text_or_data: Union[str, Dict[str, Any]], document_type: str, period: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract financial data from document

        Args:
            text_or_data: Preprocessed text from the document or data dictionary
            document_type: Type of document (Actual Income Statement, Budget, Prior Year Actual)
            period: Time period covered by the document (optional)

        Returns:
            Dict containing extracted financial data
        """
        logger.info(f"Extracting data from {document_type} document for period {period}")

        # Extract text from input
        text = self._extract_text_from_input(text_or_data)
        if not text:
            logger.warning("No text content to extract data from")
            return {'error': 'No text content to extract data from'}

        # Truncate text if too long
        max_length = 12000  # Limit to avoid token limits
        if len(text) > max_length:
            # Take first and last parts
            first_part = text[:max_length//2]
            last_part = text[-max_length//2:]
            truncated_text = first_part + "\n...[content truncated]...\n" + last_part
            logger.info(f"Text truncated from {len(text)} to {len(truncated_text)} characters")
        else:
            truncated_text = text

        try:
            # Create prompt for GPT
            prompt = self._create_extraction_prompt(truncated_text, document_type, period)
            
            # Call GPT API
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial data extraction assistant that responds only in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            extraction_result = json.loads(response_text)
            
            # Add metadata
            extraction_result['metadata'] = {
                'document_type': document_type,
                'period': period,
                'extraction_method': 'gpt-4'
            }
            
            return extraction_result
            
        except Exception as e:
            logger.error(f"Error in GPT data extraction: {str(e)}")
            return {
                'error': f"Data extraction failed: {str(e)}",
                'metadata': {
                    'document_type': document_type,
                    'period': period
                }
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
                return json.dumps(text_or_data, indent=2, default=str, ensure_ascii=False)
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

    def _create_extraction_prompt(self, text: str, document_type: str, period: Optional[str] = None) -> str:
        """
        Create prompt for GPT data extraction
        Enhanced to extract detailed income components and application fees

        Args:
            text: Preprocessed text from the document
            document_type: Type of document
            period: Time period covered by the document (optional)

        Returns:
            Prompt for GPT
        """
        # Base prompt
        prompt = f"""You are a financial data extraction assistant. Extract the following financial data from this {document_type}"""
        
        # Add period if available
        if period:
            prompt += f" for the period {period}"
            
        # Add document text
        prompt += f":\n\n{text}\n\n"
        
        # Add extraction instructions
        prompt += """Extract the following financial data points:

1. Income:
   - Gross Potential Rent (GPR)
   - Vacancy Loss
   - Concessions
   - Bad Debt/Delinquency
   - Recoveries
   - Other Income (itemized if available)
   - Application Fees (specifically look for this under Other Income)
   - Effective Gross Income (EGI) - calculate as GPR minus Vacancy/Concessions/Bad Debt plus Other Income

2. Operating Expenses:
   - Payroll
   - Administrative
   - Marketing
   - Utilities
   - Repairs & Maintenance
   - Contract Services
   - Make Ready
   - Turnover
   - Property Taxes
   - Insurance
   - Management Fees
   - Other Operating Expenses (itemized if available)
   - Total Operating Expenses

3. Reserves:
   - Replacement Reserves
   - Capital Expenditures
   - Other Reserves (itemized if available)
   - Total Reserves

4. Net Operating Income (NOI):
   - Calculate as EGI minus Total Operating Expenses

For each data point:
- Extract the exact amount from the document
- Convert all amounts to numeric values (remove currency symbols, commas, etc.)
- If a value is not found, set it to null
- If a value is negative, represent it as a negative number
- For itemized categories (like Other Income), include subcategories if available

Respond in JSON format with the following structure:
{
  "income": {
    "gross_potential_rent": number or null,
    "vacancy_loss": number or null,
    "concessions": number or null,
    "bad_debt": number or null,
    "recoveries": number or null,
    "other_income": {
      "application_fees": number or null,
      "additional_items": [
        {"name": "item name", "amount": number}
      ],
      "total": number or null
    },
    "effective_gross_income": number or null
  },
  "operating_expenses": {
    "payroll": number or null,
    "administrative": number or null,
    "marketing": number or null,
    "utilities": number or null,
    "repairs_maintenance": number or null,
    "contract_services": number or null,
    "make_ready": number or null,
    "turnover": number or null,
    "property_taxes": number or null,
    "insurance": number or null,
    "management_fees": number or null,
    "other_operating_expenses": [
      {"name": "item name", "amount": number}
    ],
    "total_operating_expenses": number or null
  },
  "reserves": {
    "replacement_reserves": number or null,
    "capital_expenditures": number or null,
    "other_reserves": [
      {"name": "item name", "amount": number}
    ],
    "total_reserves": number or null
  },
  "net_operating_income": number or null,
  "confidence_score": number between 0 and 1
}

Ensure all calculations are mathematically correct. If you're uncertain about any value, provide your best estimate and adjust the confidence score accordingly.
"""
        
        return prompt
