import os
import logging
import json
import re
import time
from typing import Dict, Any, Optional, Union
from openai import OpenAI

# Configure logging
t_logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gpt_data_extractor')

def clean_number(val: Any) -> float:
    """
    Safely convert various number formats (strings with commas, dollar signs, etc.) to float.
    """
    if isinstance(val, (int, float)):
        return float(val)
    if val is None:
        return 0.0
    s = str(val).replace(',', '').replace('$', '').strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        logger.warning(f"Unable to clean number '{val}', defaulting to 0")
        return 0.0

class GPTDataExtractor:
    """
    Class for extracting financial data from documents using GPT-4.
    """

    def __init__(self, api_key: Optional[str] = None, sample_limit: int = 3000):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.sample_limit = sample_limit

        if not self.api_key:
            logger.warning(
                "OpenAI API key not set. "
                "Please set OPENAI_API_KEY environment variable or provide it during initialization."
            )

        self.client = OpenAI(api_key=self.api_key)

    def extract(
        self,
        text_or_data: Union[str, Dict[str, Any]],
        document_type: str,
        period: Optional[str] = None
    ) -> Dict[str, Any]:
        logger.info(f"Extracting financial data from {document_type} document")
        text = self._extract_text_from_input(text_or_data)
        prompt = self._create_extraction_prompt(text, document_type, period)
        extraction_result = self._extract_with_gpt(prompt)
        self._validate_extraction_result(extraction_result, document_type)
        return extraction_result

    def _extract_text_from_input(
        self,
        text_or_data: Union[str, Dict[str, Any]]
    ) -> str:
        # [unchanged existing implementation]
        if isinstance(text_or_data, dict):
            logger.info("Input is a dictionary, extracting text content")
            if 'combined_text' in text_or_data:
                return text_or_data['combined_text']
            elif 'text' in text_or_data:
                if isinstance(text_or_data['text'], str):
                    return text_or_data['text']
                elif isinstance(text_or_data['text'], list):
                    return "\n\n".join(
                        page.get('content', '')
                        for page in text_or_data['text']
                        if isinstance(page, dict) and 'content' in page
                    )
            elif 'content' in text_or_data:
                return text_or_data['content']
            elif 'data' in text_or_data and isinstance(text_or_data['data'], str):
                return text_or_data['data']
            elif 'sheets' in text_or_data and isinstance(text_or_data['sheets'], list):
                sheet_texts = []
                for sheet in text_or_data['sheets']:
                    if isinstance(sheet, dict) and 'name' in sheet and 'data' in sheet:
                        sheet_texts.append(f"Sheet: {sheet['name']}")
                        if isinstance(sheet['data'], list):
                            for row in sheet['data']:
                                if isinstance(row, dict):
                                    sheet_texts.append(str(row))
                return "\n".join(sheet_texts)
            logger.warning(
                "Could not find text field in dictionary, using JSON string representation"
            )
            try:
                return json.dumps(text_or_data)
            except:
                return str(text_or_data)
        elif isinstance(text_or_data, str):
            return text_or_data
        else:
            logger.warning(f"Converted non-string input to string: {type(text_or_data)}")
            return str(text_or_data)

    def _create_extraction_prompt(
        self,
        text: str,
        document_type: str,
        period: Optional[str] = None
    ) -> str:
        if not isinstance(text, str):
            logger.warning("Text is not a string, converting before slicing")
            text = str(text)
        text_sample = text[:self.sample_limit]

        period_context = f" for the period {period}" if period else ""

        base_prompt = f"""
I need to extract specific financial data from this {document_type}{period_context}.

This document may include sections labeled Income, Revenue, Expenses, or Operating Expenses.
Focus on extracting these key financial metrics:

1. REVENUE ITEMS:
   - Rental Income
   - Laundry/Vending Income
   - Parking Income
   - Application Fees
   - Other Revenue
   - Total Revenue

2. EXPENSE ITEMS:
   - Repairs & Maintenance
   - Utilities
   - Property Management Fees
   - Property Taxes
   - Insurance
   - Admin/Office Costs
   - Marketing/Advertising
   - Total Expenses

3. NET OPERATING INCOME (NOI):
   - Net Operating Income (Total Revenue - Total Expenses)

Here's the financial document:
{text_sample}

Extract the financial data and provide it in JSON format with the following structure:
{{
  "document_type": "{document_type}",
  "period": "{period or 'Unknown'}",
  "rental_income": [number],
  "laundry_income": [number],
  "parking_income": [number],
  "application_fees": [number],
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

IMPORTANT:
- All values must be numbers
- Totals must equal the sum of line items
- Do not include any explanation or comments
"""
        return base_prompt

    def _extract_with_gpt(
        self,
        prompt: str
    ) -> Dict[str, Any]:
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a senior real estate accountant specializing in NOI analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            logger.info(f"GPT response time: {time.time() - start_time:.2f}s")
            response_text = response.choices[0].message.content.strip()
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                json_match = re.search(r'(\{.*\})', response_text.replace('\n', ''), re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        logger.error("Error parsing extracted JSON")
                logger.error("JSON parsing failed. Returning empty result.")
                return self._empty_result()
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return self._empty_result()

    def _validate_extraction_result(
        self,
        result: Dict[str, Any],
        document_type: str
    ) -> None:
        # Include application_fees in required_fields
        required_fields = [
            'rental_income', 'laundry_income', 'parking_income', 'other_revenue',
            'application_fees', 'total_revenue',
            'repairs_maintenance', 'utilities', 'property_management_fees', 'property_taxes',
            'insurance', 'admin_office_costs', 'marketing_advertising', 'total_expenses',
            'net_operating_income'
        ]
        for field in required_fields:
            if field not in result:
                logger.warning(f"Missing field: {field}, defaulting to 0")
                result[field] = 0
            result[field] = clean_number(result[field])

        if 'document_type' not in result:
            result['document_type'] = document_type

        # Sum revenue including application_fees
        revenue_items = [
            result['rental_income'], result['laundry_income'],
            result['parking_income'], result['application_fees'], result['other_revenue']
        ]
        # Sum expenses
        expense_items = [
            result['repairs_maintenance'], result['utilities'],
            result['property_management_fees'], result['property_taxes'],
            result['insurance'], result['admin_office_costs'], result['marketing_advertising']
        ]

        calc_revenue = sum(revenue_items)
        calc_expenses = sum(expense_items)
        calc_noi = calc_revenue - calc_expenses

        if abs(calc_revenue - result['total_revenue']) > 1:
            logger.warning(
                f"Revenue mismatch: calculated={calc_revenue}, reported={result['total_revenue']}"
            )
            result['total_revenue'] = calc_revenue

        if abs(calc_expenses - result['total_expenses']) > 1:
            logger.warning(
                f"Expenses mismatch: calculated={calc_expenses}, reported={result['total_expenses']}"
            )
            result['total_expenses'] = calc_expenses

        if abs(calc_noi - result['net_operating_income']) > 1:
            logger.warning(
                f"NOI mismatch: calculated={calc_noi}, reported={result['net_operating_income']}"
            )
            result['net_operating_income'] = calc_noi

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "document_type": "Unknown",
            "period": "Unknown",
            "rental_income": 0,
            "laundry_income": 0,
            "parking_income": 0,
            "application_fees": 0,
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

# Instantiate a singleton extractor
extractor = GPTDataExtractor()

def extract_financial_data(
    text_or_data: Union[str, Dict[str, Any]],
    document_type: str,
    period: Optional[str] = None
) -> Dict[str, Any]:
    return extractor.extract(text_or_data, document_type, period)
