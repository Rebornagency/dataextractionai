"""
Enhanced GPT Data Extraction Module for Real Estate NOI Analyzer
Uses GPT-4 to extract and structure key financial fields from preprocessed documents
with improved prompts and better handling of document types
"""

import os
import logging
import json
import re
from typing import Dict, Any, List, Optional
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gpt_data_extractor')

class GPTDataExtractor:
    """
    Class for extracting financial data from preprocessed documents using GPT-4
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the GPT data extractor
        
        Args:
            api_key: OpenAI API key (optional, can be set via environment variable)
        """
        # Set API key if provided, otherwise use environment variable
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            
        if not openai.api_key:
            logger.warning("OpenAI API key not set. Please set OPENAI_API_KEY environment variable or provide it during initialization.")
    
    def extract_data(self, text: str, document_type: str, period: str) -> Dict[str, Any]:
        """
        Extract financial data from preprocessed text using GPT-4
        
        Args:
            text: Preprocessed text from the document
            document_type: Type of document (Actual, Budget, etc.)
            period: Time period of the document (e.g., "March 2025")
            
        Returns:
            Dict containing extracted financial data
        """
        logger.info(f"Extracting financial data from {document_type} document for period {period}")
        
        # Prepare the text for GPT (limit to a reasonable size)
        # GPT-4 has a context window limit, so we need to be careful with large documents
        max_text_length = 15000  # Adjust based on your needs and model limitations
        if len(text) > max_text_length:
            logger.info(f"Text is too long ({len(text)} chars), truncating to {max_text_length} chars")
            text = text[:max_text_length]
        
        # Create the prompt for GPT using the enhanced template
        prompt = self._create_enhanced_extraction_prompt(text, document_type, period)
        
        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a senior real estate accountant specializing in NOI analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for more deterministic results
                max_tokens=1000
            )
            
            # Extract response text
            response_text = response.choices[0].message.content.strip()
            
            # Parse the JSON response
            try:
                # Find JSON in the response (in case GPT adds explanatory text)
                json_match = re.search(r'({[\s\S]*})', response_text)
                if json_match:
                    json_str = json_match.group(1)
                    extracted_data = json.loads(json_str)
                else:
                    extracted_data = json.loads(response_text)
                
                # Add document_type and period if not already included
                if 'document_type' not in extracted_data:
                    extracted_data['document_type'] = document_type
                if 'period' not in extracted_data:
                    extracted_data['period'] = period
                
                return extracted_data
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {str(e)}")
                logger.error(f"Response text: {response_text}")
                
                # Attempt to extract structured data even if JSON parsing fails
                return self._fallback_extraction(response_text, document_type, period)
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            # Return empty result with document_type and period
            return {
                'document_type': document_type,
                'period': period,
                'error': str(e)
            }
    
    def _create_enhanced_extraction_prompt(self, text: str, document_type: str, period: str) -> str:
        """
        Create an enhanced prompt for GPT-4 with more context and guidance
        
        Args:
            text: Preprocessed text from the document
            document_type: Type of document (Actual, Budget, etc.)
            period: Time period of the document
            
        Returns:
            Formatted prompt for GPT-4
        """
        prompt = f"""You are a senior real estate accountant analyzing a {document_type} for the period {period}. 

Your task is to extract specific financial data from the following real estate financial report. Focus on identifying these key financial metrics:

1. REVENUE ITEMS:
   - Rental Income: The primary income from property rentals
   - Laundry/Vending Income: Income from laundry facilities or vending machines
   - Parking Income: Revenue from parking spaces or garages
   - Other Revenue: Any additional income sources
   - Total Revenue: The sum of all revenue items

2. EXPENSE ITEMS:
   - Repairs & Maintenance: Costs for property upkeep and repairs
   - Utilities: Expenses for electricity, water, gas, etc.
   - Property Management Fees: Fees paid to property managers
   - Property Taxes: Tax expenses related to the property
   - Insurance: Property insurance costs
   - Admin/Office Costs: Administrative expenses
   - Marketing/Advertising: Costs for marketing the property
   - Total Expenses: The sum of all expense items

3. NET OPERATING INCOME (NOI):
   - This should be the difference between Total Revenue and Total Expenses

IMPORTANT GUIDELINES:
- Look for both the itemized values and the totals/subtotals
- If you find conflicting values, use the most detailed or recent figures
- Convert all values to numbers (not strings)
- If a value is not found, use null
- Ensure mathematical consistency (e.g., Total Revenue should equal the sum of revenue items)
- Pay special attention to whether values are monthly, quarterly, or annual figures

Here is the financial report text:
"""
        prompt += f'"""\n{text}\n"""'
        
        prompt += """

Return your response in this JSON format:
{
  "document_type": "Actual Income Statement",
  "period": "March 2025",
  "rental_income": 82000,
  "laundry_income": 1200,
  "parking_income": 1800,
  "other_revenue": 500,
  "total_revenue": 85500,
  "repairs_maintenance": 4300,
  "utilities": 3200,
  "property_management_fees": 2500,
  "property_taxes": 5000,
  "insurance": 1800,
  "admin_office_costs": 1200,
  "marketing_advertising": 800,
  "total_expenses": 18800,
  "net_operating_income": 66700
}

If a value is not found in the text, use null instead of a number. Make sure all financial values are numbers, not strings. Verify that total_revenue equals the sum of revenue items, total_expenses equals the sum of expense items, and net_operating_income equals total_revenue minus total_expenses.
"""
        return prompt
    
    def _fallback_extraction(self, response_text: str, document_type: str, period: str) -> Dict[str, Any]:
        """
        Fallback method to extract data when JSON parsing fails
        
        Args:
            response_text: Text response from GPT
            document_type: Type of document
            period: Time period of the document
            
        Returns:
            Dict containing extracted financial data
        """
        logger.info("Using fallback extraction method")
        
        # Initialize result with document_type and period
        result = {
            'document_type': document_type,
            'period': period
        }
        
        # Define patterns for each field
        patterns = {
            'rental_income': r'rental_income"?\s*:\s*(\d+(?:\.\d+)?)',
            'laundry_income': r'laundry_income"?\s*:\s*(\d+(?:\.\d+)?)',
            'parking_income': r'parking_income"?\s*:\s*(\d+(?:\.\d+)?)',
            'other_revenue': r'other_revenue"?\s*:\s*(\d+(?:\.\d+)?)',
            'total_revenue': r'total_revenue"?\s*:\s*(\d+(?:\.\d+)?)',
            'repairs_maintenance': r'repairs_maintenance"?\s*:\s*(\d+(?:\.\d+)?)',
            'utilities': r'utilities"?\s*:\s*(\d+(?:\.\d+)?)',
            'property_management_fees': r'property_management_fees"?\s*:\s*(\d+(?:\.\d+)?)',
            'property_taxes': r'property_taxes"?\s*:\s*(\d+(?:\.\d+)?)',
            'insurance': r'insurance"?\s*:\s*(\d+(?:\.\d+)?)',
            'admin_office_costs': r'admin_office_costs"?\s*:\s*(\d+(?:\.\d+)?)',
            'marketing_advertising': r'marketing_advertising"?\s*:\s*(\d+(?:\.\d+)?)',
            'total_expenses': r'total_expenses"?\s*:\s*(\d+(?:\.\d+)?)',
            'net_operating_income': r'net_operating_income"?\s*:\s*(\d+(?:\.\d+)?)'
        }
        
        # Extract each field using regex
        for field, pattern in patterns.items():
            match = re.search(pattern, response_text)
            if match:
                try:
                    result[field] = float(match.group(1))
                except ValueError:
                    result[field] = None
            else:
                result[field] = None
        
        return result


def extract_financial_data(text: str, document_type: str, period: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to extract financial data from a document
    
    Args:
        text: Preprocessed text from the document
        document_type: Type of document (Actual, Budget, etc.)
        period: Time period of the document
        api_key: OpenAI API key (optional)
        
    Returns:
        Dict containing extracted financial data
    """
    extractor = GPTDataExtractor(api_key)
    return extractor.extract_data(text, document_type, period)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python gpt_data_extractor.py <text_file> [document_type] [period]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    document_type = sys.argv[2] if len(sys.argv) > 2 else "Unknown"
    period = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        with open(file_path, 'r') as f:
            text = f.read()
        
        result = extract_financial_data(text, document_type, period)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
