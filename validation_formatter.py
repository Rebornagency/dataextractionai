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
        if 'income' in formatted_data:
            formatted_data['income'] = _validate_income(formatted_data['income'])
            
        # Validate and format operating expenses
        if 'operating_expenses' in formatted_data:
            formatted_data['operating_expenses'] = _validate_operating_expenses(formatted_data['operating_expenses'])
            
        # Validate and format reserves
        if 'reserves' in formatted_data:
            formatted_data['reserves'] = _validate_reserves(formatted_data['reserves'])
            
        # Validate and format NOI
        formatted_data['net_operating_income'] = _validate_noi(
            formatted_data.get('net_operating_income'),
            formatted_data.get('income', {}).get('effective_gross_income'),
            formatted_data.get('operating_expenses', {}).get('total_operating_expenses')
        )
        
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

def _validate_income(income_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and format income data
    
    Args:
        income_data: Raw income data
        
    Returns:
        Validated and formatted income data
    """
    if not income_data or not isinstance(income_data, dict):
        return {
            'gross_potential_rent': None,
            'vacancy_loss': None,
            'concessions': None,
            'bad_debt': None,
            'recoveries': None,
            'other_income': {
                'application_fees': None,
                'additional_items': [],
                'total': None
            },
            'effective_gross_income': None
        }
        
    # Ensure all fields are present
    validated_income = {
        'gross_potential_rent': _validate_numeric(income_data.get('gross_potential_rent')),
        'vacancy_loss': _validate_numeric(income_data.get('vacancy_loss')),
        'concessions': _validate_numeric(income_data.get('concessions')),
        'bad_debt': _validate_numeric(income_data.get('bad_debt')),
        'recoveries': _validate_numeric(income_data.get('recoveries')),
        'other_income': _validate_other_income(income_data.get('other_income', {})),
        'effective_gross_income': _validate_numeric(income_data.get('effective_gross_income'))
    }
    
    # Calculate EGI if not provided or validate the provided value
    calculated_egi = _calculate_egi(
        validated_income['gross_potential_rent'],
        validated_income['vacancy_loss'],
        validated_income['concessions'],
        validated_income['bad_debt'],
        validated_income['recoveries'],
        validated_income['other_income']['total']
    )
    
    if calculated_egi is not None:
        if validated_income['effective_gross_income'] is None:
            validated_income['effective_gross_income'] = calculated_egi
        else:
            # Check if the provided EGI is close to the calculated EGI
            if not _is_close(validated_income['effective_gross_income'], calculated_egi, 0.05):
                logger.warning(f"Provided EGI ({validated_income['effective_gross_income']}) differs from calculated EGI ({calculated_egi})")
                # Use the calculated value as it's more reliable
                validated_income['effective_gross_income'] = calculated_egi
                
    return validated_income

def _validate_other_income(other_income_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and format other income data
    
    Args:
        other_income_data: Raw other income data
        
    Returns:
        Validated and formatted other income data
    """
    if not other_income_data or not isinstance(other_income_data, dict):
        return {
            'application_fees': None,
            'additional_items': [],
            'total': None
        }
        
    # Validate application fees
    application_fees = _validate_numeric(other_income_data.get('application_fees'))
    
    # Validate additional items
    additional_items = []
    if 'additional_items' in other_income_data and isinstance(other_income_data['additional_items'], list):
        for item in other_income_data['additional_items']:
            if isinstance(item, dict) and 'name' in item and 'amount' in item:
                validated_item = {
                    'name': str(item['name']),
                    'amount': _validate_numeric(item['amount'])
                }
                additional_items.append(validated_item)
                
    # Validate total
    total = _validate_numeric(other_income_data.get('total'))
    
    # Calculate total if not provided or validate the provided value
    calculated_total = application_fees
    for item in additional_items:
        if item['amount'] is not None:
            if calculated_total is None:
                calculated_total = item['amount']
            else:
                calculated_total += item['amount']
                
    if calculated_total is not None:
        if total is None:
            total = calculated_total
        else:
            # Check if the provided total is close to the calculated total
            if not _is_close(total, calculated_total, 0.05):
                logger.warning(f"Provided other income total ({total}) differs from calculated total ({calculated_total})")
                # Use the calculated value as it's more reliable
                total = calculated_total
                
    return {
        'application_fees': application_fees,
        'additional_items': additional_items,
        'total': total
    }

def _validate_operating_expenses(expenses_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and format operating expenses data
    
    Args:
        expenses_data: Raw operating expenses data
        
    Returns:
        Validated and formatted operating expenses data
    """
    if not expenses_data or not isinstance(expenses_data, dict):
        return {
            'payroll': None,
            'administrative': None,
            'marketing': None,
            'utilities': None,
            'repairs_maintenance': None,
            'contract_services': None,
            'make_ready': None,
            'turnover': None,
            'property_taxes': None,
            'insurance': None,
            'management_fees': None,
            'other_operating_expenses': [],
            'total_operating_expenses': None
        }
        
    # Validate standard expense categories
    validated_expenses = {
        'payroll': _validate_numeric(expenses_data.get('payroll')),
        'administrative': _validate_numeric(expenses_data.get('administrative')),
        'marketing': _validate_numeric(expenses_data.get('marketing')),
        'utilities': _validate_numeric(expenses_data.get('utilities')),
        'repairs_maintenance': _validate_numeric(expenses_data.get('repairs_maintenance')),
        'contract_services': _validate_numeric(expenses_data.get('contract_services')),
        'make_ready': _validate_numeric(expenses_data.get('make_ready')),
        'turnover': _validate_numeric(expenses_data.get('turnover')),
        'property_taxes': _validate_numeric(expenses_data.get('property_taxes')),
        'insurance': _validate_numeric(expenses_data.get('insurance')),
        'management_fees': _validate_numeric(expenses_data.get('management_fees')),
        'other_operating_expenses': [],
        'total_operating_expenses': _validate_numeric(expenses_data.get('total_operating_expenses'))
    }
    
    # Validate other operating expenses
    if 'other_operating_expenses' in expenses_data and isinstance(expenses_data['other_operating_expenses'], list):
        for item in expenses_data['other_operating_expenses']:
            if isinstance(item, dict) and 'name' in item and 'amount' in item:
                validated_item = {
                    'name': str(item['name']),
                    'amount': _validate_numeric(item['amount'])
                }
                validated_expenses['other_operating_expenses'].append(validated_item)
                
    # Calculate total if not provided or validate the provided value
    calculated_total = None
    for key, value in validated_expenses.items():
        if key != 'other_operating_expenses' and key != 'total_operating_expenses' and value is not None:
            if calculated_total is None:
                calculated_total = value
            else:
                calculated_total += value
                
    # Add other operating expenses to the total
    for item in validated_expenses['other_operating_expenses']:
        if item['amount'] is not None:
            if calculated_total is None:
                calculated_total = item['amount']
            else:
                calculated_total += item['amount']
                
    if calculated_total is not None:
        if validated_expenses['total_operating_expenses'] is None:
            validated_expenses['total_operating_expenses'] = calculated_total
        else:
            # Check if the provided total is close to the calculated total
            if not _is_close(validated_expenses['total_operating_expenses'], calculated_total, 0.05):
                logger.warning(f"Provided operating expenses total ({validated_expenses['total_operating_expenses']}) differs from calculated total ({calculated_total})")
                # Use the calculated value as it's more reliable
                validated_expenses['total_operating_expenses'] = calculated_total
                
    return validated_expenses

def _validate_reserves(reserves_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and format reserves data
    
    Args:
        reserves_data: Raw reserves data
        
    Returns:
        Validated and formatted reserves data
    """
    if not reserves_data or not isinstance(reserves_data, dict):
        return {
            'replacement_reserves': None,
            'capital_expenditures': None,
            'other_reserves': [],
            'total_reserves': None
        }
        
    # Validate standard reserve categories
    validated_reserves = {
        'replacement_reserves': _validate_numeric(reserves_data.get('replacement_reserves')),
        'capital_expenditures': _validate_numeric(reserves_data.get('capital_expenditures')),
        'other_reserves': [],
        'total_reserves': _validate_numeric(reserves_data.get('total_reserves'))
    }
    
    # Validate other reserves
    if 'other_reserves' in reserves_data and isinstance(reserves_data['other_reserves'], list):
        for item in reserves_data['other_reserves']:
            if isinstance(item, dict) and 'name' in item and 'amount' in item:
                validated_item = {
                    'name': str(item['name']),
                    'amount': _validate_numeric(item['amount'])
                }
                validated_reserves['other_reserves'].append(validated_item)
                
    # Calculate total if not provided or validate the provided value
    calculated_total = None
    if validated_reserves['replacement_reserves'] is not None:
        calculated_total = validated_reserves['replacement_reserves']
        
    if validated_reserves['capital_expenditures'] is not None:
        if calculated_total is None:
            calculated_total = validated_reserves['capital_expenditures']
        else:
            calculated_total += validated_reserves['capital_expenditures']
            
    # Add other reserves to the total
    for item in validated_reserves['other_reserves']:
        if item['amount'] is not None:
            if calculated_total is None:
                calculated_total = item['amount']
            else:
                calculated_total += item['amount']
                
    if calculated_total is not None:
        if validated_reserves['total_reserves'] is None:
            validated_reserves['total_reserves'] = calculated_total
        else:
            # Check if the provided total is close to the calculated total
            if not _is_close(validated_reserves['total_reserves'], calculated_total, 0.05):
                logger.warning(f"Provided reserves total ({validated_reserves['total_reserves']}) differs from calculated total ({calculated_total})")
                # Use the calculated value as it's more reliable
                validated_reserves['total_reserves'] = calculated_total
                
    return validated_reserves

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
