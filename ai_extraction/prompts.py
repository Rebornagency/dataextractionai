"""
Prompt templates for different document types to use with GPT extraction.
These provide document-specific extraction instructions to improve accuracy.
"""

PROMPT_TEMPLATES = {
    "actual": """Extract the following line items from this Actual P&L PDF:
    - gross_potential_rent
    - vacancy_loss
    - other_income
    - operating_expenses
    Return ONLY a valid JSON object with these keys. Do not explain.""",
    
    "budget": """Extract the following line items from this Budget P&L PDF:
    - gross_potential_rent
    - vacancy_loss
    - other_income
    - operating_expenses
    Return ONLY a valid JSON object with these keys. These are budgeted values.""",
    
    "pro_forma": """Extract the following line items from this Pro Forma P&L PDF:
    - gross_potential_rent
    - vacancy_loss
    - other_income
    - operating_expenses
    Return ONLY a valid JSON object with these keys. These are pro forma projections.""",
    
    "t12": """Extract the following line items from this T12 P&L PDF:
    - gross_potential_rent
    - vacancy_loss
    - other_income
    - operating_expenses
    Return ONLY a valid JSON object with these keys. These represent trailing 12 month values.""",
    
    "rent_roll": """Extract the following line items from this Rent Roll PDF:
    - number_of_units
    - occupied_units
    - vacancy_percentage
    - total_monthly_rent
    Return ONLY a valid JSON object with these keys. Do not explain.""",
    
    "operating_statement": """Extract the following line items from this Operating Statement PDF:
    - gross_potential_rent
    - vacancy_loss
    - other_income
    - operating_expenses
    - net_operating_income
    Return ONLY a valid JSON object with these keys. Do not explain."""
}

# Default prompt to use if document type is not in the templates
DEFAULT_PROMPT = """Extract the following financial data points from this document:
- gross_potential_rent
- vacancy_loss
- other_income
- operating_expenses 
- net_operating_income
Return ONLY a valid JSON object with these keys. Do not explain.""" 