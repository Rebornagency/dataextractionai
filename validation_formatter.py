import logging
import math
from typing import Dict, Any, List, Optional, Union
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('validation_formatter')

def validate_and_format_data(extraction_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and format the extracted data
    
    Args:
        extraction_result: Raw data extracted from GPT
        
    Returns:
        Validated and formatted data
    """
    logger.info("Validating and formatting extracted data")
    
    # Check if extraction result is valid
    if not extraction_result or not isinstance(extraction_result, dict):
        logger.error("Invalid extraction result")
        return {
            'error': 'Invalid extraction result',
            'data': None
        }
        
    # Check if there was an error in extraction
    if 'error' in extraction_result:
        logger.error(f"Error in extraction result: {extraction_result['error']}")
        return extraction_result
        
    # Create a copy of the extraction result to avoid modifying the original
    formatted_data = extraction_result.copy()
    
    try:
        # Validate and format income data
        formatted_data = _validate_income_fields(formatted_data)
            
        # Validate and format operating expenses
        formatted_data = _validate_operating_expenses_fields(formatted_data)
            
        # Validate and format reserves
        formatted_data = _validate_reserves_fields(formatted_data)
            
        # Validate and format NOI
        formatted_data['net_operating_income'] = _validate_noi(
            formatted_data.get('net_operating_income'),
            formatted_data.get('effective_gross_income'),
            formatted_data.get('operating_expenses', {}).get('total_operating_expenses')
        )
        
        # Validate property_id and period
        formatted_data['property_id'] = formatted_data.get('property_id')
        formatted_data['period'] = _validate_period(formatted_data.get('period'))
        
        # Perform additional validation checks
        formatted_data = _perform_validation_checks(formatted_data)
        
        # Add validation status
        formatted_data['validation'] = {
            'status': 'valid',
            'message': 'Data validated and formatted successfully'
        }
        
    except Exception as e:
        logger.error(f"Error validating and formatting data: {str(e)}")
        formatted_data['validation'] = {
            'status': 'error',
            'message': f"Error validating and formatting data: {str(e)}"
        }
        
    return formatted_data

def _validate_income_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and format income fields
    
    Args:
        data: Raw data
        
    Returns:
        Validated and formatted data
    """
    # Extract income fields from flattened structure
    gross_potential_rent = _validate_numeric(data.get('gross_potential_rent'))
    vacancy_loss = _validate_numeric(data.get('vacancy_loss'))
    concessions = _validate_numeric(data.get('concessions'))
    bad_debt = _validate_numeric(data.get('bad_debt'))
    recoveries = _validate_numeric(data.get('recoveries'))
    
    # Handle other_income which is still a nested structure
    other_income = data.get('other_income', {})
    if not isinstance(other_income, dict):
        other_income = {}
    
    application_fees = _validate_numeric(other_income.get('application_fees'))
    additional_items = []
    if 'additional_items' in other_income and isinstance(other_income['additional_items'], list):
        for item in other_income['additional_items']:
            if isinstance(item, dict) and 'name' in item and 'amount' in item:
                validated_item = {
                    'name': str(item['name']),
                    'amount': _validate_numeric(item['amount'])
                }
                additional_items.append(validated_item)
    
    other_income_total = _validate_numeric(other_income.get('total'))
    
    # Calculate other_income total if not provided
    calculated_other_income = application_fees
    for item in additional_items:
        if item['amount'] is not None:
            if calculated_other_income is None:
                calculated_other_income = item['amount']
            else:
                calculated_other_income += item['amount']
    
    if calculated_other_income is not None:
        if other_income_total is None:
            other_income_total = calculated_other_income
        elif not _is_close(other_income_total, calculated_other_income, 0.05):
            logger.warning(f"Provided other income total ({other_income_total}) differs from calculated total ({calculated_other_income})")
            other_income_total = calculated_other_income
    
    # Validate EGI
    effective_gross_income = _validate_numeric(data.get('effective_gross_income'))
    
    # Calculate EGI if not provided
    calculated_egi = _calculate_egi(
        gross_potential_rent,
        vacancy_loss,
        concessions,
        bad_debt,
        recoveries,
        other_income_total
    )
    
    if calculated_egi is not None:
        if effective_gross_income is None:
            effective_gross_income = calculated_egi
        elif not _is_close(effective_gross_income, calculated_egi, 0.05):
            logger.warning(f"Provided EGI ({effective_gross_income}) differs from calculated EGI ({calculated_egi})")
            effective_gross_income = calculated_egi
    
    # Update data with validated values
    data['gross_potential_rent'] = gross_potential_rent
    data['vacancy_loss'] = vacancy_loss
    data['concessions'] = concessions
    data['bad_debt'] = bad_debt
    data['recoveries'] = recoveries
    data['other_income'] = {
        'application_fees': application_fees,
        'additional_items': additional_items,
        'total': other_income_total
    }
    data['effective_gross_income'] = effective_gross_income
    
    return data

def _validate_operating_expenses_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and format operating expenses fields
    
    Args:
        data: Raw data
        
    Returns:
        Validated and formatted data
    """
    # Extract operating expenses from nested structure
    operating_expenses = data.get('operating_expenses', {})
    if not isinstance(operating_expenses, dict):
        operating_expenses = {}
    
    # Validate standard expense categories
    payroll = _validate_numeric(operating_expenses.get('payroll'))
    administrative = _validate_numeric(operating_expenses.get('administrative'))
    marketing = _validate_numeric(operating_expenses.get('marketing'))
    utilities = _validate_numeric(operating_expenses.get('utilities'))
    repairs_maintenance = _validate_numeric(operating_expenses.get('repairs_maintenance'))
    contract_services = _validate_numeric(operating_expenses.get('contract_services'))
    make_ready = _validate_numeric(operating_expenses.get('make_ready'))
    turnover = _validate_numeric(operating_expenses.get('turnover'))
    property_taxes = _validate_numeric(operating_expenses.get('property_taxes'))
    insurance = _validate_numeric(operating_expenses.get('insurance'))
    management_fees = _validate_numeric(operating_expenses.get('management_fees'))
    
    # Validate other operating expenses
    other_operating_expenses = []
    if 'other_operating_expenses' in operating_expenses and isinstance(operating_expenses['other_operating_expenses'], list):
        for item in operating_expenses['other_operating_expenses']:
            if isinstance(item, dict) and 'name' in item and 'amount' in item:
                validated_item = {
                    'name': str(item['name']),
                    'amount': _validate_numeric(item['amount'])
                }
                other_operating_expenses.append(validated_item)
    
    # Validate total operating expenses
    total_operating_expenses = _validate_numeric(operating_expenses.get('total_operating_expenses'))
    
    # Calculate total if not provided
    calculated_total = None
    for value in [payroll, administrative, marketing, utilities, repairs_maintenance, 
                 contract_services, make_ready, turnover, property_taxes, insurance, management_fees]:
        if value is not None:
            if calculated_total is None:
                calculated_total = value
            else:
                calculated_total += value
    
    # Add other operating expenses to the total
    for item in other_operating_expenses:
        if item['amount'] is not None:
            if calculated_total is None:
                calculated_total = item['amount']
            else:
                calculated_total += item['amount']
    
    if calculated_total is not None:
        if total_operating_expenses is None:
            total_operating_expenses = calculated_total
        elif not _is_close(total_operating_expenses, calculated_total, 0.05):
            logger.warning(f"Provided operating expenses total ({total_operating_expenses}) differs from calculated total ({calculated_total})")
            total_operating_expenses = calculated_total
    
    # Update data with validated values
    data['operating_expenses'] = {
        'payroll': payroll,
        'administrative': administrative,
        'marketing': marketing,
        'utilities': utilities,
        'repairs_maintenance': repairs_maintenance,
        'contract_services': contract_services,
        'make_ready': make_ready,
        'turnover': turnover,
        'property_taxes': property_taxes,
        'insurance': insurance,
        'management_fees': management_fees,
        'other_operating_expenses': other_operating_expenses,
        'total_operating_expenses': total_operating_expenses
    }
    
    # Also add total_operating_expenses to top level for NOI Analyzer
    data['operating_expenses_total'] = total_operating_expenses
    
    return data

def _validate_reserves_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and format reserves fields
    
    Args:
        data: Raw data
        
    Returns:
        Validated and formatted data
    """
    # Extract reserves from nested structure
    reserves = data.get('reserves', {})
    if not isinstance(reserves, dict):
        reserves = {}
    
    # Validate standard reserve categories
    replacement_reserves = _validate_numeric(reserves.get('replacement_reserves'))
    capital_expenditures = _validate_numeric(reserves.get('capital_expenditures'))
    
    # Validate other reserves
    other_reserves = []
    if 'other_reserves' in reserves and isinstance(reserves['other_reserves'], list):
        for item in reserves['other_reserves']:
            if isinstance(item, dict) and 'name' in item and 'amount' in item:
                validated_item = {
                    'name': str(item['name']),
                    'amount': _validate_numeric(item['amount'])
                }
                other_reserves.append(validated_item)
    
    # Validate total reserves
    total_reserves = _validate_numeric(reserves.get('total_reserves'))
    
    # Calculate total if not provided
    calculated_total = None
    if replacement_reserves is not None:
        calculated_total = replacement_reserves
    
    if capital_expenditures is not None:
        if calculated_total is None:
            calculated_total = capital_expenditures
        else:
            calculated_total += capital_expenditures
    
    # Add other reserves to the total
    for item in other_reserves:
        if item['amount'] is not None:
            if calculated_total is None:
                calculated_total = item['amount']
            else:
                calculated_total += item['amount']
    
    if calculated_total is not None:
        if total_reserves is None:
            total_reserves = calculated_total
        elif not _is_close(total_reserves, calculated_total, 0.05):
            logger.warning(f"Provided reserves total ({total_reserves}) differs from calculated total ({calculated_total})")
            total_reserves = calculated_total
    
    # Update data with validated values
    data['reserves'] = {
        'replacement_reserves': replacement_reserves,
        'capital_expenditures': capital_expenditures,
        'other_reserves': other_reserves,
        'total_reserves': total_reserves
    }
    
    return data

def _perform_validation_checks(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform additional validation checks
    
    Args:
        data: Data to validate
        
    Returns:
        Validated data
    """
    # 1. Non-negative Values Check
    for field in ['gross_potential_rent', 'vacancy_loss', 'concessions', 'bad_debt', 'operating_expenses_total', 'net_operating_income']:
        value = data.get(field)
        if value is not None and value < 0:
            logger.warning(f"Negative value found for {field}: {value}. Setting to 0.")
            data[field] = 0.0
    
    # 2. EGI Sanity Check
    gpr = data.get('gross_potential_rent', 0) or 0
    vacancy_loss = data.get('vacancy_loss', 0) or 0
    concessions = data.get('concessions', 0) or 0
    bad_debt = data.get('bad_debt', 0) or 0
    other_income_total = data.get('other_income', {}).get('total', 0) or 0
    
    calculated_egi = gpr - (vacancy_loss + concessions + bad_debt) + other_income_total
    
    if calculated_egi < 0:
        logger.warning(f"Calculated EGI is negative: {calculated_egi}. Adjusting vacancy/concessions/bad_debt.")
        # Reset problematic values to 0
        data['vacancy_loss'] = 0.0
        data['concessions'] = 0.0
        data['bad_debt'] = 0.0
        # Recalculate EGI
        data['effective_gross_income'] = gpr + other_income_total
    
    # 3. NOI Consistency Check
    egi = data.get('effective_gross_income', 0) or 0
    opex = data.get('operating_expenses_total', 0) or 0
    noi = data.get('net_operating_income', 0) or 0
    
    calculated_noi = egi - opex
    
    if noi is not None and calculated_noi is not None:
        if not _is_close(noi, calculated_noi, 0.01):
            logger.warning(f"NOI consistency check failed: provided NOI ({noi}) differs from calculated NOI ({calculated_noi})")
            data['net_operating_income'] = calculated_noi
    
    # Set fallback values for non-required fields
    if data.get('concessions') is None:
        data['concessions'] = 0.0
    
    if data.get('bad_debt') is None:
        data['bad_debt'] = 0.0
    
    return data

def _validate_period(period: Optional[str]) -> Optional[str]:
    """
    Validate and format period string to YYYY-MM format
    
    Args:
        period: Period string
        
    Returns:
        Validated period string
    """
    if not period:
        return None
    
    # Try to extract YYYY-MM format
    import re
    
    # Check if already in YYYY-MM format
    if re.match(r'^\d{4}-\d{2}$', period):
        return period
    
    # Try to extract year and month from various formats
    year_month_match = re.search(r'(\d{4})[-/\s]*(\d{1,2})', period)
    if year_month_match:
        year = year_month_match.group(1)
        month = year_month_match.group(2).zfill(2)  # Ensure month is 2 digits
        return f"{year}-{month}"
    
    # Try to extract from month name formats (e.g., "January 2025")
    month_names = {
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05', 'jun': '06',
        'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
    }
    
    month_year_match = re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,.-]+(\d{4})', 
                                period.lower())
    if month_year_match:
        month_abbr = month_year_match.group(1)
        year = month_year_match.group(2)
        return f"{year}-{month_names[month_abbr]}"
    
    # If we can't parse it, return as is
    logger.warning(f"Could not parse period '{period}' into YYYY-MM format")
    return period

def _validate_noi(noi: Optional[Union[float, int, str]], egi: Optional[Union[float, int, str]], opex: Optional[Union[float, int, str]]) -> Optional[float]:
    """
    Validate and format NOI
    
    Args:
        noi: Raw NOI value
        egi: Effective Gross Income
        opex: Total Operating Expenses
        
    Returns:
        Validated and formatted NOI
    """
    # Validate provided NOI
    validated_noi = _validate_numeric(noi)
    
    # Calculate NOI if EGI and OpEx are available
    validated_egi = _validate_numeric(egi)
    validated_opex = _validate_numeric(opex)
    
    if validated_egi is not None and validated_opex is not None:
        calculated_noi = validated_egi - validated_opex
        
        if validated_noi is None:
            return calculated_noi
        else:
            # Check if the provided NOI is close to the calculated NOI
            if not _is_close(validated_noi, calculated_noi, 0.05):
                logger.warning(f"Provided NOI ({validated_noi}) differs from calculated NOI ({calculated_noi})")
                # Use the calculated value as it's more reliable
                return calculated_noi
                
    return validated_noi

def _validate_numeric(value: Optional[Union[float, int, str]]) -> Optional[float]:
    """
    Validate and convert a value to a numeric type
    
    Args:
        value: Value to validate
        
    Returns:
        Validated numeric value or None
    """
    if value is None:
        return None
        
    # If value is already a number, validate it
    if isinstance(value, (int, float)):
        # Check for NaN or infinity
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
        
    # If value is a string, try to convert it
    if isinstance(value, str):
        # Remove currency symbols, commas, and handle parentheses for negative numbers
        cleaned_value = value.replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
        try:
            return float(cleaned_value)
        except ValueError:
            return None
            
    # If value is any other type, return None
    return None

def _calculate_egi(gpr: Optional[float], vacancy: Optional[float], concessions: Optional[float], 
                  bad_debt: Optional[float], recoveries: Optional[float], other_income: Optional[float]) -> Optional[float]:
    """
    Calculate Effective Gross Income
    
    Args:
        gpr: Gross Potential Rent
        vacancy: Vacancy Loss
        concessions: Concessions
        bad_debt: Bad Debt
        recoveries: Recoveries
        other_income: Other Income Total
        
    Returns:
        Calculated EGI or None if not enough data
    """
    # Need at least GPR to calculate EGI
    if gpr is None:
        return None
        
    # Start with GPR
    egi = gpr
    
    # Subtract vacancy loss if available
    if vacancy is not None:
        egi -= vacancy
        
    # Subtract concessions if available
    if concessions is not None:
        egi -= concessions
        
    # Subtract bad debt if available
    if bad_debt is not None:
        egi -= bad_debt
        
    # Add recoveries if available
    if recoveries is not None:
        egi += recoveries
        
    # Add other income if available
    if other_income is not None:
        egi += other_income
        
    return egi

def _is_close(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
    """
    Check if two values are close to each other
    
    Args:
        a: First value
        b: Second value
        rel_tol: Relative tolerance
        abs_tol: Absolute tolerance
        
    Returns:
        True if values are close, False otherwise
    """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
