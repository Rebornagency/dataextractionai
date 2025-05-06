"""
Prompt templates for different document types to use with GPT extraction.
These provide document-specific extraction instructions to improve accuracy.
"""

PROMPT_TEMPLATES = {
    "actual": """Extract the following line items from this Actual P&L PDF:
    - gross_potential_rent
    - vacancy_loss
    - other_income (including DETAILED breakdown of parking, laundry, late_fees, pet_fees, application_fees, and any other income items)
    - operating_expenses (including DETAILED breakdown of property_taxes, insurance, repairs_maintenance, utilities, management_fees, and any other operating expense line items)
    Return ONLY a valid JSON object with these keys. Do not explain.
    
    For other_income, please return a nested structure with all individual income components AND the total:
    "other_income": {
        "parking": 2100,
        "laundry": 1500,
        "late_fees": 900,
        "pet_fees": 400,
        "application_fees": 300,
        ... other income items if present ...
        "total": 5200
    }
    
    For operating expenses, please return a nested structure with all individual expense components AND the total:
    "operating_expenses": {
        "property_taxes": 12345.67,
        "insurance": 2345.67,
        "repairs_maintenance": 3456.78,
        "utilities": 4567.89, 
        "management_fees": 567.89,
        ... other expense items if present ...
        "total_operating_expenses": 30000
    }""",
    
    "budget": """Extract the following line items from this Budget P&L PDF:
    - gross_potential_rent
    - vacancy_loss
    - other_income (including DETAILED breakdown of parking, laundry, late_fees, pet_fees, application_fees, and any other income items)
    - operating_expenses (including DETAILED breakdown of property_taxes, insurance, repairs_maintenance, utilities, management_fees, and any other operating expense line items)
    Return ONLY a valid JSON object with these keys. These are budgeted values.
    
    For other_income, please return a nested structure with all individual income components AND the total:
    "other_income": {
        "parking": 2100,
        "laundry": 1500,
        "late_fees": 900,
        "pet_fees": 400,
        "application_fees": 300,
        ... other income items if present ...
        "total": 5200
    }
    
    For operating expenses, please return a nested structure with all individual expense components AND the total:
    "operating_expenses": {
        "property_taxes": 12345.67,
        "insurance": 2345.67,
        "repairs_maintenance": 3456.78,
        "utilities": 4567.89,
        "management_fees": 567.89,
        ... other expense items if present ...
        "total_operating_expenses": 30000
    }""",
    
    "pro_forma": """Extract the following line items from this Pro Forma P&L PDF:
    - gross_potential_rent
    - vacancy_loss
    - other_income (including DETAILED breakdown of parking, laundry, late_fees, pet_fees, application_fees, and any other income items)
    - operating_expenses (including DETAILED breakdown of property_taxes, insurance, repairs_maintenance, utilities, management_fees, and any other operating expense line items)
    Return ONLY a valid JSON object with these keys. These are pro forma projections.
    
    For other_income, please return a nested structure with all individual income components AND the total:
    "other_income": {
        "parking": 2100,
        "laundry": 1500,
        "late_fees": 900,
        "pet_fees": 400,
        "application_fees": 300,
        ... other income items if present ...
        "total": 5200
    }
    
    For operating expenses, please return a nested structure with all individual expense components AND the total:
    "operating_expenses": {
        "property_taxes": 12345.67,
        "insurance": 2345.67,
        "repairs_maintenance": 3456.78,
        "utilities": 4567.89,
        "management_fees": 567.89,
        ... other expense items if present ...
        "total_operating_expenses": 30000
    }""",
    
    "t12": """Extract the following line items from this T12 P&L PDF:
    - gross_potential_rent
    - vacancy_loss
    - other_income (including DETAILED breakdown of parking, laundry, late_fees, pet_fees, application_fees, and any other income items)
    - operating_expenses (including DETAILED breakdown of property_taxes, insurance, repairs_maintenance, utilities, management_fees, and any other operating expense line items)
    Return ONLY a valid JSON object with these keys. These represent trailing 12 month values.
    
    For other_income, please return a nested structure with all individual income components AND the total:
    "other_income": {
        "parking": 2100,
        "laundry": 1500,
        "late_fees": 900,
        "pet_fees": 400,
        "application_fees": 300,
        ... other income items if present ...
        "total": 5200
    }
    
    For operating expenses, please return a nested structure with all individual expense components AND the total:
    "operating_expenses": {
        "property_taxes": 12345.67,
        "insurance": 2345.67,
        "repairs_maintenance": 3456.78,
        "utilities": 4567.89,
        "management_fees": 567.89,
        ... other expense items if present ...
        "total_operating_expenses": 30000
    }""",
    
    "rent_roll": """Extract the following line items from this Rent Roll PDF:
    - number_of_units
    - occupied_units
    - vacancy_percentage
    - total_monthly_rent
    Return ONLY a valid JSON object with these keys. Do not explain.""",
    
    "operating_statement": """Extract the following line items from this Operating Statement PDF:
    - gross_potential_rent
    - vacancy_loss
    - other_income (including DETAILED breakdown of parking, laundry, late_fees, pet_fees, application_fees, and any other income items)
    - operating_expenses (including DETAILED breakdown of property_taxes, insurance, repairs_maintenance, utilities, management_fees, and any other operating expense line items)
    - net_operating_income
    Return ONLY a valid JSON object with these keys. Do not explain.
    
    For other_income, please return a nested structure with all individual income components AND the total:
    "other_income": {
        "parking": 2100,
        "laundry": 1500,
        "late_fees": 900,
        "pet_fees": 400,
        "application_fees": 300,
        ... other income items if present ...
        "total": 5200
    }
    
    For operating expenses, please return a nested structure with all individual expense components AND the total:
    "operating_expenses": {
        "property_taxes": 12345.67,
        "insurance": 2345.67,
        "repairs_maintenance": 3456.78,
        "utilities": 4567.89,
        "management_fees": 567.89,
        ... other expense items if present ...
        "total_operating_expenses": 30000
    }"""
}

# Default prompt to use if document type is not in the templates
DEFAULT_PROMPT = """Extract the following financial data points from this document:
- gross_potential_rent
- vacancy_loss
- other_income (including DETAILED breakdown of parking, laundry, late_fees, pet_fees, application_fees, and any other income items)
- operating_expenses (including DETAILED breakdown of property_taxes, insurance, repairs_maintenance, utilities, management_fees, and any other operating expense line items)
- net_operating_income
Return ONLY a valid JSON object with these keys. Do not explain.

For other_income, please return a nested structure with all individual income components AND the total:
"other_income": {
    "parking": 2100,
    "laundry": 1500,
    "late_fees": 900,
    "pet_fees": 400,
    "application_fees": 300,
    ... other income items if present ...
    "total": 5200
}

For operating expenses, please return a nested structure with all individual expense components AND the total:
"operating_expenses": {
    "property_taxes": 12345.67,
    "insurance": 2345.67,
    "repairs_maintenance": 3456.78,
    "utilities": 4567.89,
    "management_fees": 567.89,
    ... other expense items if present ...
    "total_operating_expenses": 30000
}""" 