"""
Enhanced GPT Data Extractor Module for NOI Analyzer
This module is updated to work with the new clearly labeled document approach
and properly utilize document type information from the NOI Tool
"""

import os
import logging
import json
import re
from typing import Dict, Any, List, Optional, Union
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gpt_data_extractor')

class GPTDataExtractor:
    """
    Class for extracting financial data from documents using GPT-4
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the GPT data extractor
        
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
    
    def extract(self, text_or_data: Union[str, Dict[str, Any]], document_type: str, period: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract financial data from document
        
        Args:
            text_or_data: Preprocessed text from the document or data dictionary
            document_type: Type of document
            period: Time period of the document (optional)
            
        Returns:
            Dict containing extracted financial data
        """
        logger.info(f"Extracting financial data from {document_type} document")
        
        # Extract text from input
        text = self._extract_text_from_input(text_or_data)
        
        # Create extraction prompt
        prompt = self._create_extraction_prompt(text, document_type, period)
        
        # Extract data using GPT
        extraction_result = self._extract_with_gpt(prompt)
        
        # Validate extraction result
        self._validate_extraction_result(extraction_result, document_type)
        
        return extraction_result
    
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
            logger.info("Input is a dictionary, extracting text content")
            
            # Try to extract text from common dictionary structures
            if 'combined_text' in text_or_data:
                return text_or_data['combined_text']
            elif 'text' in text_or_data:
                if isinstance(text_or_data['text'], str):
                    return text_or_data['text']
                elif isinstance(text_or_data['text'], list) and len(text_or_data['text']) > 0:
                    # Join text from multiple pages
                    return "\n\n".join([page.get('content', '') for page in text_or_data['text'] if isinstance(page, dict) and 'content' in page])
            elif 'content' in text_or_data:
                return text_or_data['content']
            elif 'data' in text_or_data and isinstance(text_or_data['data'], str):
                return text_or_data['data']
            elif 'sheets' in text_or_data and isinstance(text_or_data['sheets'], list):
                # Extract text from Excel sheets
                sheet_texts = []
                for sheet in text_or_data['sheets']:
                    if isinstance(sheet, dict) and 'name' in sheet and 'data' in sheet:
                        sheet_texts.append(f"Sheet: {sheet['name']}")
                        if isinstance(sheet['data'], list):
                            for row in sheet['data']:
                                if isinstance(row, dict):
                                    sheet_texts.append(str(row))
                return "\n".join(sheet_texts)
            
            # If we couldn't extract text using known structures, convert the entire dictionary to a string
            logger.warning("Could not find text field in dictionary, using JSON string representation")
            try:
                return json.dumps(text_or_data)
            except:
                return str(text_or_data)
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
        Create extraction prompt based on document type and period
        
        Args:
            text: Preprocessed text from the document
            document_type: Type of document
            period: Time period of the document (optional)
            
        Returns:
            Extraction prompt for GPT
        """
        # Prepare a sample of the text for GPT (first 3000 characters)
        text_sample = text[:3000]
        
        # Create document type specific context
        doc_type_context = ""
        if document_type == "Actual Income Statement":
            doc_type_context = """This is an Actual Income Statement showing real financial results for a specific property.

Important notes for Actual Income Statements:
- These are ACTUAL results, not projections or budgets
- Look for sections labeled "Income", "Revenue", "Expenses", or "Operating Expenses"
- The document may use terms like "YTD", "Month-to-Date", or specific month names
- Focus on the most recent or relevant period if multiple periods are shown
- Ensure mathematical consistency in your extraction"""
            
        elif document_type == "Budget":
            doc_type_context = """This is a Budget document showing projected financial figures for a specific property.

Important notes for Budget documents:
- These are PROJECTED figures, not actual results
- Look for sections labeled "Budget", "Forecast", "Plan", or "Projected"
- The document may include comparisons to actual results
- Focus on the budget figures, not the actual or variance columns
- Ensure mathematical consistency in your extraction"""
            
        elif document_type == "Prior Year Actual":
            doc_type_context = """This is a Prior Year Actual statement showing historical financial results for a specific property.

Important notes for Prior Year Actual statements:
- These are HISTORICAL results from a previous period
- Look for sections labeled with previous year indicators
- The document may include comparisons to current year or budget
- Focus on the prior year figures, not current year or budget columns
- Ensure mathematical consistency in your extraction"""
        
        # Add period context if available
        period_context = f" for the period {period}" if period else ""
        
        # Create the base prompt
        base_prompt = f"""I need to extract specific financial data from this {document_type}{period_context}.

{doc_type_context}

Focus on extracting these key financial metrics:

1. REVENUE ITEMS:
   - Rental Income: The primary income from property rentals (may be called "Rent Income", "Rental Revenue", etc.)
   - Laundry/Vending Income: Income from laundry facilities or vending machines (may be called "Laundry", "Vending", "Other Income - Laundry", etc.)
   - Parking Income: Revenue from parking spaces or garages (may be called "Parking", "Garage Income", "Parking Fees", etc.)
   - Other Revenue: Any additional income sources (may include late fees, application fees, pet fees, etc.)
   - Total Revenue: The sum of all revenue items (may be called "Total Income", "Gross Income", "Total Revenue", etc.)

2. EXPENSE ITEMS:
   - Repairs & Maintenance: Costs for property upkeep and repairs (may include general maintenance, cleaning, landscaping, etc.)
   - Utilities: Expenses for electricity, water, gas, etc. (may be broken down by utility type)
   - Property Management Fees: Fees paid to property managers (may be called "Management Fee", "Property Management", etc.)
   - Property Taxes: Tax expenses related to the property (may be called "Real Estate Taxes", "Property Tax", etc.)
   - Insurance: Property insurance costs (may be called "Insurance Expense", "Property Insurance", etc.)
   - Admin/Office Costs: Administrative expenses (may include office supplies, software, professional fees, etc.)
   - Marketing/Advertising: Costs for marketing the property (may be called "Advertising", "Marketing Expense", etc.)
   - Total Expenses: The sum of all expense items (may be called "Total Operating Expenses", "Total Expenses", etc.)

3. NET OPERATING INCOME (NOI):
   - This should be the difference between Total Revenue and Total Expenses (may be called "NOI", "Net Operating Income", "Operating Income", etc.)

Here's the financial document:
{text_sample}

Extract the financial data and provide it in JSON format with the following structure:
{{
  "document_type": "{document_type}",
  "period": "{period or 'Unknown'}",
  "rental_income": [number],
  "laundry_income": [number],
  "parking_income": [number],
  "other_revenue": [number],
  "total_revenue": [number],
  "repairs_maintenance": [number],
  "utilities": [number],
  "property_management_fees": [number],
  "property_taxes": [number],
  "insurance": [number],
  "admin_office_costs": [number],
  "marketing_advertising": [number],
  "total_expenses": [number],
  "net_operating_income": [number]
}}

IMPORTANT REQUIREMENTS:
1. All financial values MUST be numbers, not strings
2. The total_revenue MUST equal the sum of revenue items
3. The total_expenses MUST equal the sum of expense items
4. The net_operating_income MUST equal total_revenue minus total_expenses
5. If a value is not found, use 0 rather than leaving it blank
6. Do not include any explanations or notes in your response, only the JSON object
"""
        
        return base_prompt
    
    def _extract_with_gpt(self, prompt: str) -> Dict[str, Any]:
        """
        Extract data using GPT-4
        
        Args:
            prompt: Extraction prompt
            
        Returns:
            Dict containing extracted financial data
        """
        try:
            # Call OpenAI API with updated client format
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a senior real estate accountant specializing in NOI analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for more deterministic results
                max_tokens=800
            )
            
            # Extract response text
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
                logger.info("Successfully parsed JSON response")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {str(e)}")
                logger.error(f"Response text: {response_text}")
                
                # Try to extract JSON from response text
                json_match = re.search(r'({.*})', response_text.replace('\n', ''), re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                        logger.info("Successfully extracted and parsed JSON from response")
                        return result
                    except json.JSONDecodeError:
                        logger.error("Error parsing extracted JSON")
                
                # Return empty result if JSON parsing fails
                return {
                    "document_type": "Unknown",
                    "period": "Unknown",
                    "rental_income": 0,
                    "laundry_income": 0,
                    "parking_income": 0,
                    "other_revenue": 0,
                    "total_revenue": 0,
                    "repairs_maintenance": 0,
                    "utilities": 0,
                    "property_management_fees": 0,
                    "property_taxes": 0,
                    "insurance": 0,
                    "admin_office_costs": 0,
                    "marketing_advertising": 0,
                    "total_expenses": 0,
                    "net_operating_income": 0
                }
                
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            # Return empty result if API call fails
            return {
                "document_type": "Unknown",
                "period": "Unknown",
                "rental_income": 0,
                "laundry_income": 0,
                "parking_income": 0,
                "other_revenue": 0,
                "total_revenue": 0,
                "repairs_maintenance": 0,
                "utilities": 0,
                "property_management_fees": 0,
                "property_taxes": 0,
                "insurance": 0,
                "admin_office_costs": 0,
                "marketing_advertising": 0,
                "total_expenses": 0,
                "net_operating_income": 0
            }
    
    def _validate_extraction_result(self, result: Dict[str, Any], document_type: str) -> None:
        """
        Validate extraction result and ensure it has the required fields
        
        Args:
            result: Extraction result
            document_type: Type of document
        """
        # Required fields for all document types
        required_fields = [
            'rental_income',
            'laundry_income',
            'parking_income',
            'other_revenue',
            'total_revenue',
            'repairs_maintenance',
            'utilities',
            'property_management_fees',
            'property_taxes',
            'insurance',
            'admin_office_costs',
            'marketing_advertising',
            'total_expenses',
            'net_operating_income'
        ]
        
        # Check for missing fields and set them to 0
        for field in required_fields:
            if field not in result:
                logger.warning(f"Missing field in extraction result: {field}")
                result[field] = 0
            
            # Ensure numeric values
            try:
                result[field] = float(result[field])
            except (ValueError, TypeError):
                logger.warning(f"Non-numeric value for {field}: {result[field]}")
                result[field] = 0
        
        # Add document type if missing
        if 'document_type' not in result:
            result['document_type'] = document_type
            
        # Verify mathematical consistency
        revenue_items = [
            result.get('rental_income', 0),
            result.get('laundry_income', 0),
            result.get('parking_income', 0),
            result.get('other_revenue', 0)
        ]
        
        expense_items = [
            result.get('repairs_maintenance', 0),
            result.get('utilities', 0),
            result.get('property_management_fees', 0),
            result.get('property_taxes', 0),
            result.get('insurance', 0),
            result.get('admin_office_costs', 0),
            result.get('marketing_advertising', 0)
        ]
        
        # Calculate sums
        calculated_total_revenue = sum(revenue_items)
        calculated_total_expenses = sum(expense_items)
        calculated_noi = calculated_total_revenue - calculated_total_expenses
        
        # Check for significant discrepancies (more than $1 difference)
        if abs(calculated_total_revenue - result.get('total_revenue', 0)) > 1:
            logger.warning(f"Total revenue discrepancy: calculated {calculated_total_revenue}, reported {result.get('total_revenue', 0)}")
            # Use calculated value for consistency
            result['total_revenue'] = calculated_total_revenue
            
        if abs(calculated_total_expenses - result.get('total_expenses', 0)) > 1:
            logger.warning(f"Total expenses discrepancy: calculated {calculated_total_expenses}, reported {result.get('total_expenses', 0)}")
            # Use calculated value for consistency
            result['total_expenses'] = calculated_total_expenses
            
        if abs(calculated_noi - result.get('net_operating_income', 0)) > 1:
            logger.warning(f"NOI discrepancy: calculated {calculated_noi}, reported {result.get('net_operating_income', 0)}")
            # Use calculated value for consistency
            result['net_operating_income'] = calculated_noi

# Create an instance of the extractor for direct import
extractor = GPTDataExtractor()

# Function to extract financial data (for backward compatibility)
def extract_financial_data(text_or_data: Union[str, Dict[str, Any]], document_type: str, period: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract financial data from document
    
    Args:
        text_or_data: Preprocessed text from the document or data dictionary
        document_type: Type of document
        period: Time period of the document (optional)
        
    Returns:
        Dict containing extracted financial data
    """
    return extractor.extract(text_or_data, document_type, period)
