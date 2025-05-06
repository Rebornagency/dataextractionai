import logging
import json
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('helpers')

def format_for_noi_comparison(api_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format API response for NOI comparison.
    
    Args:
        api_response: Response from extraction API
        
    Returns:
        Formatted data for NOI comparison
    """
    logger.info("Formatting API response for NOI comparison")
    logger.debug(f"API response keys: {list(api_response.keys() if isinstance(api_response, dict) else [])}")
    
    # Initialize result with default values
    result = {
        'property_id': None,
        'period': None,
        'gpr': None,
        'vacancy_loss': None,
        'concessions': None,
        'bad_debt': None,
        'other_income': None,
        'egi': None,
        'opex': None,
        'noi': None,
        # Add OpEx breakdown components
        'property_taxes': None,
        'insurance': None,
        'repairs_and_maintenance': None,
        'utilities': None,
        'management_fees': None,
        # Add Other Income breakdown components
        'parking': None,
        'laundry': None,
        'late_fees': None,
        'pet_fees': None,
        'application_fees': None
    }
    
    # Return empty result if API response is invalid
    if not api_response or not isinstance(api_response, dict):
        logger.error(f"Invalid API response: {api_response}")
        return result
    
    # Extract property_id and period
    result['property_id'] = api_response.get('property_id')
    result['period'] = api_response.get('period')
    
    # Extract financial data - handle both nested and flat structures
    financials = api_response.get('financials', {})
    
    # If financials is not a dict, try to use the top-level response
    if not isinstance(financials, dict) or not financials:
        logger.warning("No 'financials' object found in API response, using top-level fields")
        financials = api_response
    
    # Log the structure for debugging
    logger.debug(f"Financials structure: {list(financials.keys() if isinstance(financials, dict) else [])}")
    
    # Helper function to safely extract numeric values
    def safe_float(value):
        if value is None:
            return None
        
        # Handle numpy types
        if hasattr(value, 'item'):
            try:
                return float(value.item())
            except:
                pass
        
        # Handle dictionary values (nested structure)
        if isinstance(value, dict):
            # Try to find a numeric value in the dictionary
            for key in ['amount', 'value', 'total']:
                if key in value and value[key] is not None:
                    return safe_float(value[key])
            return None
        
        # Handle string values
        if isinstance(value, str):
            # Remove currency symbols and commas
            clean_value = value.replace('$', '').replace(',', '').strip()
            try:
                return float(clean_value)
            except ValueError:
                return None
        
        # Handle numeric values
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    # Extract financial values with safe conversion
    result['gpr'] = safe_float(financials.get('gross_potential_rent'))
    result['vacancy_loss'] = safe_float(financials.get('vacancy_loss'))
    result['concessions'] = safe_float(financials.get('concessions'))
    result['bad_debt'] = safe_float(financials.get('bad_debt'))
    
    # Handle other_income which could be nested or a direct value
    other_income = financials.get('other_income')
    if isinstance(other_income, dict) and 'total' in other_income:
        result['other_income'] = safe_float(other_income['total'])
        
        # Extract Other Income components from the nested structure
        other_income_components = {
            'parking': ['parking'],
            'laundry': ['laundry'],
            'late_fees': ['late_fees', 'late fees'],
            'pet_fees': ['pet_fees', 'pet fees', 'pet rent'],
            'application_fees': ['application_fees', 'application fees']
        }
        
        # Try to extract each component using various possible field names
        for result_key, possible_fields in other_income_components.items():
            for field in possible_fields:
                if field in other_income and other_income[field] is not None:
                    result[result_key] = safe_float(other_income[field])
                    break
    else:
        result['other_income'] = safe_float(other_income)
    
    # Also check for Other Income components at the top level of financials
    # The extraction tool may put these at the top level
    other_income_keys = ['parking', 'laundry', 'late_fees', 'pet_fees', 'application_fees']
    for component in other_income_keys:
        # Only set if not already set from nested structure
        if result[component] is None and component in financials:
            result[component] = safe_float(financials[component])
    
    # Ensure all Other Income components have at least a zero value if we have other_income
    if result['other_income'] is not None and result['other_income'] > 0:
        for component in other_income_keys:
            if result[component] is None:
                result[component] = 0.0
    
    # Handle EGI (effective_gross_income)
    egi = financials.get('effective_gross_income')
    if egi is None:
        # Try total_revenue as an alternative
        egi = financials.get('total_revenue')
    result['egi'] = safe_float(egi)
    
    # Handle operating expenses
    opex = financials.get('total_expenses')
    if opex is None:
        # Try alternative fields
        opex = financials.get('operating_expenses')
        if isinstance(opex, dict) and 'total_operating_expenses' in opex:
            opex = opex['total_operating_expenses']
        elif opex is None:
            opex = financials.get('operating_expenses_total')
    result['opex'] = safe_float(opex)
    
    # Handle NOI
    noi = financials.get('net_operating_income')
    result['noi'] = safe_float(noi)
    
    # --- extract OpEx breakdown components ---
    # Try nested structure first
    opex_source = {}
    if isinstance(financials.get("operating_expenses"), dict):
        opex_source = financials["operating_expenses"]
    # Fall back to flat top-level
    else:
        opex_source = financials

    # Map new component keys to possible field names
    component_mapping = {
        "property_taxes": ["property_taxes", "taxes"],
        "insurance": ["insurance"],
        "repairs_and_maintenance": ["repairs_and_maintenance", "repairs_maintenance", "repairs", "maintenance"],
        "utilities": ["utilities"],
        "management_fees": ["management_fees", "management"]
    }

    for key, candidates in component_mapping.items():
        raw = None
        for field in candidates:
            if field in opex_source:
                raw = opex_source[field]
                break
        # Also check top-level in case extraction tool flattened them
        if raw is None and key in financials:
            raw = financials[key]
        result[key] = safe_float(raw) if raw is not None else 0.0
    
    # Calculate EGI if not provided
    if result['egi'] is None and result['gpr'] is not None:
        egi = result['gpr']
        if result['vacancy_loss'] is not None:
            egi -= result['vacancy_loss']
        if result['concessions'] is not None:
            egi -= result['concessions']
        if result['bad_debt'] is not None:
            egi -= result['bad_debt']
        if result['other_income'] is not None:
            egi += result['other_income']
        result['egi'] = egi
    
    # Calculate NOI if not provided
    if result['noi'] is None and result['egi'] is not None and result['opex'] is not None:
        result['noi'] = result['egi'] - result['opex']
    
    # Log the formatted result
    logger.info(f"Formatted result: property_id={result['property_id']}, period={result['period']}")
    logger.debug(f"Financial values: gpr={result['gpr']}, vacancy_loss={result['vacancy_loss']}, egi={result['egi']}, opex={result['opex']}, noi={result['noi']}")
    logger.debug(f"OpEx breakdown: taxes={result['property_taxes']}, insurance={result['insurance']}, r&m={result['repairs_and_maintenance']}")
    logger.debug(f"Other Income breakdown: parking={result['parking']}, laundry={result['laundry']}, late_fees={result['late_fees']}")
    
    return result

def calculate_noi_comparisons(
    current_data: Dict[str, Any],
    budget_data: Optional[Dict[str, Any]] = None,
    prior_month_data: Optional[Dict[str, Any]] = None,
    prior_year_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Calculate NOI comparisons between current data and other periods.
    
    Args:
        current_data: Current period data
        budget_data: Budget data (optional)
        prior_month_data: Prior month data (optional)
        prior_year_data: Prior year data (optional)
        
    Returns:
        Dictionary with comparison results
    """
    logger.info("Calculating NOI comparisons")
    
    # Initialize result
    result = {
        'current': current_data,
        'actual_vs_budget': None,
        'month_vs_prior': None,
        'year_vs_year': None
    }
    
    # Calculate actual vs budget comparison
    if budget_data:
        logger.info("Calculating actual vs budget comparison")
        avb = {}
        
        for key in ['gpr', 'vacancy_loss', 'concessions', 'bad_debt', 'other_income', 'egi', 'opex', 'noi']:
            current_value = current_data.get(key)
            budget_value = budget_data.get(key)
            
            # Skip if either value is None
            if current_value is None or budget_value is None:
                avb[f'{key}_budget'] = budget_value
                avb[f'{key}_variance'] = None
                avb[f'{key}_percent_variance'] = None
                continue
            
            # Calculate variance
            variance = current_value - budget_value
            
            # Calculate percent variance
            if budget_value != 0:
                percent_variance = (variance / abs(budget_value)) * 100
            else:
                percent_variance = 0 if variance == 0 else float('inf')
            
            # Store results
            avb[f'{key}_budget'] = budget_value
            avb[f'{key}_variance'] = variance
            avb[f'{key}_percent_variance'] = percent_variance
        
        result['actual_vs_budget'] = avb
    
    # Calculate month vs prior month comparison
    if prior_month_data:
        logger.info("Calculating month vs prior month comparison")
        mom = {}
        
        for key in ['gpr', 'vacancy_loss', 'concessions', 'bad_debt', 'other_income', 'egi', 'opex', 'noi']:
            current_value = current_data.get(key)
            prior_value = prior_month_data.get(key)
            
            # Skip if either value is None
            if current_value is None or prior_value is None:
                mom[f'{key}_prior'] = prior_value
                mom[f'{key}_change'] = None
                mom[f'{key}_percent_change'] = None
                continue
            
            # Calculate change
            change = current_value - prior_value
            
            # Calculate percent change
            if prior_value != 0:
                percent_change = (change / abs(prior_value)) * 100
            else:
                percent_change = 0 if change == 0 else float('inf')
            
            # Store results
            mom[f'{key}_prior'] = prior_value
            mom[f'{key}_change'] = change
            mom[f'{key}_percent_change'] = percent_change
        
        result['month_vs_prior'] = mom
    
    # Calculate year vs prior year comparison
    if prior_year_data:
        logger.info("Calculating year vs prior year comparison")
        yoy = {}
        
        for key in ['gpr', 'vacancy_loss', 'concessions', 'bad_debt', 'other_income', 'egi', 'opex', 'noi']:
            current_value = current_data.get(key)
            prior_value = prior_year_data.get(key)
            
            # Skip if either value is None
            if current_value is None or prior_value is None:
                yoy[f'{key}_prior_year'] = prior_value
                yoy[f'{key}_change'] = None
                yoy[f'{key}_percent_change'] = None
                continue
            
            # Calculate change
            change = current_value - prior_value
            
            # Calculate percent change
            if prior_value != 0:
                percent_change = (change / abs(prior_value)) * 100
            else:
                percent_change = 0 if change == 0 else float('inf')
            
            # Store results
            yoy[f'{key}_prior_year'] = prior_value
            yoy[f'{key}_change'] = change
            yoy[f'{key}_percent_change'] = percent_change
        
        result['year_vs_year'] = yoy
    
    return result

def format_currency(value: Optional[float]) -> str:
    """
    Format value as currency.
    
    Args:
        value: Value to format
        
    Returns:
        Formatted currency string
    """
    if value is None:
        return "N/A"
    return f"${value:,.2f}"

def format_percent(value: Optional[float]) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Value to format
        
    Returns:
        Formatted percentage string
    """
    if value is None:
        return "N/A"
    return f"{value:.1f}%"

def format_change(value: Optional[float], is_favorable: bool = True) -> str:
    """
    Format change value with color indicator.
    
    Args:
        value: Value to format
        is_favorable: Whether positive change is favorable
        
    Returns:
        Formatted change string with color indicator
    """
    if value is None:
        return "N/A"
    
    if value > 0:
        color = "green" if is_favorable else "red"
        return f"<span style='color:{color}'>+{format_currency(value)}</span>"
    elif value < 0:
        color = "red" if is_favorable else "green"
        return f"<span style='color:{color}'>{format_currency(value)}</span>"
    else:
        return f"{format_currency(value)}"

def format_percent_change(value: Optional[float], is_favorable: bool = True) -> str:
    """
    Format percentage change with color indicator.
    
    Args:
        value: Value to format
        is_favorable: Whether positive change is favorable
        
    Returns:
        Formatted percentage change string with color indicator
    """
    if value is None:
        return "N/A"
    
    if value > 0:
        color = "green" if is_favorable else "red"
        return f"<span style='color:{color}'>+{format_percent(value)}</span>"
    elif value < 0:
        color = "red" if is_favorable else "green"
        return f"<span style='color:{color}'>{format_percent(value)}</span>"
    else:
        return f"{format_percent(value)}"

def create_comparison_dataframe(comparison_data: Dict[str, Any], comparison_type: str) -> pd.DataFrame:
    """
    Create DataFrame for comparison visualization.
    
    Args:
        comparison_data: Comparison data
        comparison_type: Type of comparison (actual_vs_budget, month_vs_prior, year_vs_year)
        
    Returns:
        DataFrame with comparison data
    """
    logger.info(f"Creating comparison DataFrame for {comparison_type}")
    
    # Get current data
    current = comparison_data.get('current', {})
    
    # Get comparison data based on type
    if comparison_type == 'actual_vs_budget':
        comp = comparison_data.get('actual_vs_budget', {})
        if not comp:
            return pd.DataFrame()
        
        # Create DataFrame
        data = []
        for key, label in [
            ('gpr', 'Gross Potential Rent'),
            ('vacancy_loss', 'Vacancy Loss'),
            ('other_income', 'Other Income'),
            ('egi', 'Effective Gross Income'),
            ('opex', 'Operating Expenses'),
            ('noi', 'Net Operating Income')
        ]:
            # Determine if positive variance is favorable (different for expenses)
            is_favorable = key not in ['vacancy_loss', 'opex']
            
            # Get values
            actual = current.get(key)
            budget = comp.get(f'{key}_budget')
            variance = comp.get(f'{key}_variance')
            percent_variance = comp.get(f'{key}_percent_variance')
            
            # Format values
            actual_fmt = format_currency(actual)
            budget_fmt = format_currency(budget)
            variance_fmt = format_change(variance, is_favorable)
            percent_variance_fmt = format_percent_change(percent_variance, is_favorable)
            
            # Add to data
            data.append({
                'Category': label,
                'Actual': actual_fmt,
                'Budget': budget_fmt,
                'Variance': variance_fmt,
                'Variance %': percent_variance_fmt,
                'Actual_raw': actual,
                'Budget_raw': budget,
                'Variance_raw': variance,
                'Variance_%_raw': percent_variance
            })
        
        return pd.DataFrame(data)
    
    elif comparison_type == 'month_vs_prior':
        comp = comparison_data.get('month_vs_prior', {})
        if not comp:
            return pd.DataFrame()
        
        # Create DataFrame
        data = []
        for key, label in [
            ('gpr', 'Gross Potential Rent'),
            ('vacancy_loss', 'Vacancy Loss'),
            ('other_income', 'Other Income'),
            ('egi', 'Effective Gross Income'),
            ('opex', 'Operating Expenses'),
            ('noi', 'Net Operating Income')
        ]:
            # Determine if positive change is favorable (different for expenses)
            is_favorable = key not in ['vacancy_loss', 'opex']
            
            # Get values
            current_value = current.get(key)
            prior = comp.get(f'{key}_prior')
            change = comp.get(f'{key}_change')
            percent_change = comp.get(f'{key}_percent_change')
            
            # Format values
            current_fmt = format_currency(current_value)
            prior_fmt = format_currency(prior)
            change_fmt = format_change(change, is_favorable)
            percent_change_fmt = format_percent_change(percent_change, is_favorable)
            
            # Add to data
            data.append({
                'Category': label,
                'Current': current_fmt,
                'Prior': prior_fmt,
                'Change': change_fmt,
                'Change %': percent_change_fmt,
                'Current_raw': current_value,
                'Prior_raw': prior,
                'Change_raw': change,
                'Change_%_raw': percent_change
            })
        
        return pd.DataFrame(data)
    
    elif comparison_type == 'year_vs_year':
        comp = comparison_data.get('year_vs_year', {})
        if not comp:
            return pd.DataFrame()
        
        # Create DataFrame
        data = []
        for key, label in [
            ('gpr', 'Gross Potential Rent'),
            ('vacancy_loss', 'Vacancy Loss'),
            ('other_income', 'Other Income'),
            ('egi', 'Effective Gross Income'),
            ('opex', 'Operating Expenses'),
            ('noi', 'Net Operating Income')
        ]:
            # Determine if positive change is favorable (different for expenses)
            is_favorable = key not in ['vacancy_loss', 'opex']
            
            # Get values
            current_value = current.get(key)
            prior = comp.get(f'{key}_prior_year')
            change = comp.get(f'{key}_change')
            percent_change = comp.get(f'{key}_percent_change')
            
            # Format values
            current_fmt = format_currency(current_value)
            prior_fmt = format_currency(prior)
            change_fmt = format_change(change, is_favorable)
            percent_change_fmt = format_percent_change(percent_change, is_favorable)
            
            # Add to data
            data.append({
                'Category': label,
                'Current': current_fmt,
                'Prior Year': prior_fmt,
                'Change': change_fmt,
                'Change %': percent_change_fmt,
                'Current_raw': current_value,
                'Prior_Year_raw': prior,
                'Change_raw': change,
                'Change_%_raw': percent_change
            })
        
        return pd.DataFrame(data)
    
    # Return empty DataFrame if comparison type is not recognized
    return pd.DataFrame()

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
