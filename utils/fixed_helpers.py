"""
Helper functions for the NOI Analyzer application
"""

import logging
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('helpers')

def format_for_noi_comparison(api_response: Dict[str, Any]) -> Dict[str, float]:
    """
    Format financial data for NOI comparison calculations

    Args:
        api_response: Complete API response containing nested financial data

    Returns:
        Formatted data with standardized fields expected by noi_calculations.py
    """
    # Initialize result with default values for all expected keys
    formatted_data = {
        "gpr": 0.0,               # Gross Potential Rent
        "vacancy_loss": 0.0,      # Vacancy & Credit Loss
        "other_income": 0.0,      # Other Income
        "egi": 0.0,               # Effective Gross Income
        "opex": 0.0,              # Operating Expenses
        "noi": 0.0                # Net Operating Income
    }

    # Log the input structure for debugging
    logger.info(f"Transforming financial data: API response -> calculation format")
    logger.info(f"Input keys: {list(api_response.keys())}")
    
    # Handle both nested and flat structures
    financials = {}
    
    # Check if we have a nested financials object
    if 'financials' in api_response and isinstance(api_response['financials'], dict):
        logger.info("Found nested 'financials' object in API response")
        financials = api_response['financials']
    else:
        # Handle flat structure (original format)
        logger.info("No 'financials' object found, using flat structure")
        financials = {
            'gross_potential_rent': api_response.get('gross_potential_rent'),
            'vacancy_loss': api_response.get('vacancy_loss'),
            'concessions': api_response.get('concessions'),
            'bad_debt': api_response.get('bad_debt'),
            'other_income': api_response.get('other_income'),
            'total_revenue': api_response.get('egi') or api_response.get('effective_gross_income'),
            'total_expenses': api_response.get('operating_expenses'),
            'net_operating_income': api_response.get('noi') or api_response.get('net_operating_income')
        }
    
    logger.info(f"Financials object keys: {list(financials.keys())}")
    
    # Extract the relevant values with defaults of 0 for missing values
    revenue = financials.get('total_revenue', 0)
    expenses = financials.get('total_expenses', 0)
    noi = financials.get('net_operating_income', 0)
    
    # Extract additional values if available
    gpr = financials.get('gross_potential_rent', revenue)  # Default to revenue if not provided
    vacancy_loss = financials.get('vacancy_loss', 0)
    other_income = financials.get('other_income', 0)
    
    # Handle None values and nested dictionaries
    # For revenue
    if isinstance(revenue, dict):
        logger.info(f"Revenue is a dictionary: {revenue}")
        # Try to extract a numeric value from the dictionary
        if 'total' in revenue:
            revenue = revenue.get('total', 0)
        elif 'amount' in revenue:
            revenue = revenue.get('amount', 0)
        else:
            # If no clear numeric field, use the first numeric value found
            for key, value in revenue.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    revenue = value
                    break
            else:
                revenue = 0.0
    revenue = 0.0 if revenue is None else float(revenue)
    
    # For expenses
    if isinstance(expenses, dict):
        logger.info(f"Expenses is a dictionary: {expenses}")
        # Try to extract a numeric value from the dictionary
        if 'total' in expenses:
            expenses = expenses.get('total', 0)
        elif 'total_operating_expenses' in expenses:
            expenses = expenses.get('total_operating_expenses', 0)
        elif 'amount' in expenses:
            expenses = expenses.get('amount', 0)
        else:
            # If no clear numeric field, use the first numeric value found
            for key, value in expenses.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    expenses = value
                    break
            else:
                expenses = 0.0
    expenses = 0.0 if expenses is None else float(expenses)
    
    # For NOI
    if isinstance(noi, dict):
        logger.info(f"NOI is a dictionary: {noi}")
        # Try to extract a numeric value from the dictionary
        if 'total' in noi:
            noi = noi.get('total', 0)
        elif 'amount' in noi:
            noi = noi.get('amount', 0)
        else:
            # If no clear numeric field, use the first numeric value found
            for key, value in noi.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    noi = value
                    break
            else:
                noi = 0.0
    noi = 0.0 if noi is None else float(noi)
    
    # For GPR
    if isinstance(gpr, dict):
        logger.info(f"GPR is a dictionary: {gpr}")
        # Try to extract a numeric value from the dictionary
        if 'total' in gpr:
            gpr = gpr.get('total', 0)
        elif 'amount' in gpr:
            gpr = gpr.get('amount', 0)
        else:
            # If no clear numeric field, use the first numeric value found
            for key, value in gpr.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    gpr = value
                    break
            else:
                gpr = 0.0
    gpr = 0.0 if gpr is None else float(gpr)
    
    # For vacancy loss
    if isinstance(vacancy_loss, dict):
        logger.info(f"Vacancy loss is a dictionary: {vacancy_loss}")
        # Try to extract a numeric value from the dictionary
        if 'total' in vacancy_loss:
            vacancy_loss = vacancy_loss.get('total', 0)
        elif 'amount' in vacancy_loss:
            vacancy_loss = vacancy_loss.get('amount', 0)
        else:
            # If no clear numeric field, use the first numeric value found
            for key, value in vacancy_loss.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    vacancy_loss = value
                    break
            else:
                vacancy_loss = 0.0
    vacancy_loss = 0.0 if vacancy_loss is None else float(vacancy_loss)
    
    # For other income
    if isinstance(other_income, dict):
        logger.info(f"Other income is a dictionary: {other_income}")
        # Try to extract a numeric value from the dictionary
        if 'total' in other_income:
            other_income = other_income.get('total', 0)
        elif 'amount' in other_income:
            other_income = other_income.get('amount', 0)
        else:
            # If no clear numeric field, use the first numeric value found
            for key, value in other_income.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    other_income = value
                    break
            else:
                other_income = 0.0
    other_income = 0.0 if other_income is None else float(other_income)
    
    # Calculate EGI if not directly provided
    egi = revenue  # Default EGI to revenue
    
    # If NOI is not provided but we have revenue and expenses, calculate it
    if noi == 0 and revenue != 0 and expenses != 0:
        noi = revenue - expenses

    # Update the formatted data with all required fields
    formatted_data["gpr"] = gpr
    formatted_data["vacancy_loss"] = vacancy_loss
    formatted_data["other_income"] = other_income
    formatted_data["egi"] = egi
    formatted_data["opex"] = expenses
    formatted_data["noi"] = noi
    
    # Log the values for debugging
    logger.info(f"Extracted values: revenue={revenue}, expenses={expenses}, noi={noi}")
    logger.info(f"Output keys: {list(formatted_data.keys())}")
    logger.info(f"Output values: gpr={formatted_data['gpr']}, opex={formatted_data['opex']}, noi={formatted_data['noi']}")

    return formatted_data

def determine_document_type(filename: str, result: Dict[str, Any]) -> str:
    """
    Determine the document type based on filename and content

    Args:
        filename: Name of the file
        result: Extraction result

    Returns:
        Document type (current_month, prior_month, budget, prior_year)
    """
    filename = filename.lower()

    # Try to determine from filename first
    if "budget" in filename:
        return "budget"
    elif "prior" in filename or "previous" in filename:
        if "year" in filename:
            return "prior_year"
        else:
            return "prior_month"
    elif "current" in filename or "actual" in filename:
        return "current_month"

    # If not determined from filename, try to use document_type from result
    doc_type = result.get("document_type", "").lower()
    if "budget" in doc_type:
        return "budget"
    elif "prior year" in doc_type or "previous year" in doc_type:
        return "prior_year"
    elif "prior" in doc_type or "previous" in doc_type:
        return "prior_month"
    elif "current" in doc_type or "actual" in doc_type:
        return "current_month"

    # Default to current_month if can't determine
    return "current_month"

def format_currency(value: float) -> str:
    """
    Format a value as currency

    Args:
        value: Numeric value to format

    Returns:
        Formatted currency string
    """
    return f"${value:,.2f}"

def format_percent(value: float) -> str:
    """
    Format a value as percentage

    Args:
        value: Numeric value to format

    Returns:
        Formatted percentage string
    """
    return f"{value:.2f}%"
