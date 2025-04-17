"""
Enhanced GPT Data Extractor Module for NOI Analyzer - Part 1
This module is updated to work with the new clearly labeled document approach
and properly utilize document type information from the NOI Tool
"""

import os
import logging
import json
import re
from typing import Dict, Any, List, Tuple, Optional, Union
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gpt_data_extractor')

class GPTDataExtractor:
    """
    Class for extracting financial data from documents using GPT-4,
    enhanced to work with known document types from the NOI Tool
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
"""
Enhanced GPT Data Extractor Module for NOI Analyzer - Part 2
This module is updated to work with the new clearly labeled document approach
and properly utilize document type information from the NOI Tool
"""

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
"""
Enhanced GPT Data Extractor Module for NOI Analyzer - Part 3
This module is updated to work with the new clearly labeled document approach
and properly utilize document type information from the NOI Tool
"""

def _get_system_prompt(self, document_type: str) -> str:
    """
    Get system prompt based on document type
    
    Args:
        document_type: Type of document
        
    Returns:
        System prompt for GPT
    """
    if document_type == "Actual Income Statement":
        return """You are a senior real estate financial analyst specializing in NOI (Net Operating Income) analysis. 
Your task is to extract financial data from actual income statements with precision and accuracy.

You understand that:
1. Actual income statements show real financial results for a specific period
2. Numbers should be treated as exact values, not estimates or projections
3. The NOI calculation must be mathematically consistent (total_revenue - total_expenses = net_operating_income)
4. All financial values must be extracted as numbers, not text
5. You must identify all revenue and expense categories even if they use slightly different terminology

Your expertise in real estate accounting allows you to recognize standard financial categories and properly categorize them according to industry standards."""
        
    elif document_type == "Budget":
        return """You are a senior real estate financial analyst specializing in NOI (Net Operating Income) analysis. 
Your task is to extract financial data from budget documents with precision and accuracy.

You understand that:
1. Budget documents contain projected or planned financial figures, not actual results
2. These projections represent expected performance for a future period
3. The NOI calculation must be mathematically consistent (total_revenue - total_expenses = net_operating_income)
4. All financial values must be extracted as numbers, not text
5. You must identify all revenue and expense categories even if they use slightly different terminology

Your expertise in real estate accounting allows you to recognize standard financial categories in budget documents and properly categorize them according to industry standards."""
        
    elif document_type == "Prior Year Actual":
        return """You are a senior real estate financial analyst specializing in NOI (Net Operating Income) analysis. 
Your task is to extract financial data from prior year actual statements with precision and accuracy.

You understand that:
1. Prior year actual statements show historical financial results from a previous period
2. These figures represent past performance, not current or projected results
3. The NOI calculation must be mathematically consistent (total_revenue - total_expenses = net_operating_income)
4. All financial values must be extracted as numbers, not text
5. You must identify all revenue and expense categories even if they use slightly different terminology

Your expertise in real estate accounting allows you to recognize standard financial categories in historical statements and properly categorize them according to industry standards."""
        
    else:
        return """You are a senior real estate financial analyst specializing in NOI (Net Operating Income) analysis. 
Your task is to extract financial data from real estate financial documents with precision and accuracy.

You understand that:
1. Financial documents may contain actual results, budgets, or projections
2. The NOI calculation must be mathematically consistent (total_revenue - total_expenses = net_operating_income)
3. All financial values must be extracted as numbers, not text
4. You must identify all revenue and expense categories even if they use slightly different terminology

Your expertise in real estate accounting allows you to recognize standard financial categories and properly categorize them according to industry standards."""
"""
Enhanced GPT Data Extractor Module for NOI Analyzer - Part 4
This module is updated to work with the new clearly labeled document approach
and properly utilize document type information from the NOI Tool
"""

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
"""
Enhanced GPT Data Extractor Module for NOI Analyzer - Part 5
This module is updated to work with the new clearly labeled document approach
and properly utilize document type information from the NOI Tool
"""

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
