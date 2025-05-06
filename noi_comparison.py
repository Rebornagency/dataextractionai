import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('noi_comparison')

def calculate_noi_comparisons(current_data: Dict[str, Any], compare_data: Optional[Dict[str, Any]] = None, 
                             show_zero_values: bool = True) -> Dict[str, Any]:
    """
    Calculate NOI comparison metrics including detailed OpEx breakdown
    
    Args:
        current_data: Current period financial data
        compare_data: Prior period financial data for comparison (optional)
        show_zero_values: Whether to include zero values in the output
        
    Returns:
        Dictionary with comparison metrics
    """
    logger.info("Calculating NOI comparison metrics with OpEx breakdown")
    
    # Initialize comparison results
    comparison_results = {
        "current_data": current_data,
        "compare_data": compare_data
    }
    
    # Define all metrics to compare, including the new OpEx components
    metrics = [
        "gross_potential_rent", "vacancy_loss", "concessions", "bad_debt", "other_income",
        "parking", "laundry", "late_fees", "pet_fees", "application_fees",
        "effective_gross_income", "opex", "property_taxes", "insurance", "repairs_and_maintenance",
        "utilities", "management_fees", "noi"
    ]
    
    # Normalize current data
    normalized_current = _normalize_financials_data(current_data)
    
    # Normalize compare data if provided
    normalized_compare = None
    if compare_data:
        normalized_compare = _normalize_financials_data(compare_data)
    
    # Calculate comparisons for each metric
    for metric in metrics:
        # Get current value
        current_value = normalized_current.get(metric, 0)
        
        # Skip if current value is zero and show_zero_values is False
        if current_value == 0 and not show_zero_values and metric != "noi":
            continue
            
        # Calculate comparison metrics
        compare_value = None
        change = None
        percent_change = None
        
        if normalized_compare:
            compare_value = normalized_compare.get(metric, 0)
            
            # Calculate change
            change = current_value - compare_value
            
            # Calculate percent change
            if compare_value != 0:
                percent_change = (change / abs(compare_value)) * 100
        
        # Add to comparison results
        comparison_results[f"{metric}_current"] = current_value
        
        if compare_value is not None:
            comparison_results[f"{metric}_compare"] = compare_value
            comparison_results[f"{metric}_change"] = change
            comparison_results[f"{metric}_percent_change"] = percent_change
    
    return comparison_results

def _normalize_financials_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize financials data to ensure consistent structure
    Extracts all OpEx components from nested structures
    
    Args:
        data: Financial data to normalize
        
    Returns:
        Normalized data dictionary
    """
    # Create a copy to avoid modifying the original
    normalized = {}
    
    # Get financials object if it exists
    financials = data.get("financials", data)
    
    # Copy all top-level financials fields
    for key, value in financials.items():
        normalized[key] = value
    
    # Extract nested structures if they exist
    if "operating_expenses" in financials and isinstance(financials["operating_expenses"], dict):
        opex_data = financials["operating_expenses"]
        
        # Set total opex
        if "total_operating_expenses" in opex_data:
            normalized["opex"] = opex_data["total_operating_expenses"]
        
        # Extract each OpEx component
        component_mapping = {
            "property_taxes": ["property_taxes", "taxes"],
            "insurance": ["insurance"],
            "repairs_and_maintenance": ["repairs_maintenance", "repairs", "maintenance"],
            "utilities": ["utilities"],
            "management_fees": ["management_fees", "management"]
        }
        
        # Try to extract each component using various possible field names
        for normalized_key, possible_fields in component_mapping.items():
            for field in possible_fields:
                if field in opex_data and opex_data[field] is not None:
                    normalized[normalized_key] = opex_data[field]
                    break
    
    # Extract Other Income nested structure if it exists
    if "other_income" in financials and isinstance(financials["other_income"], dict):
        other_income_data = financials["other_income"]
        
        # Set total other income
        if "total" in other_income_data:
            normalized["other_income"] = other_income_data["total"]
        
        # Extract each Other Income component
        component_mapping = {
            "parking": ["parking"],
            "laundry": ["laundry"],
            "late_fees": ["late_fees", "late fees"],
            "pet_fees": ["pet_fees", "pet fees", "pet rent"],
            "application_fees": ["application_fees", "application fees"]
        }
        
        # Try to extract each component using various possible field names
        for normalized_key, possible_fields in component_mapping.items():
            for field in possible_fields:
                if field in other_income_data and other_income_data[field] is not None:
                    normalized[normalized_key] = other_income_data[field]
                    break
    
    # Handle legacy/flat structure where components might be at the top level
    if "opex" not in normalized and "operating_expenses_total" in financials:
        normalized["opex"] = financials["operating_expenses_total"]
    
    # Extract OpEx components that might be at the top level
    top_level_components = [
        "property_taxes", "insurance", "repairs_and_maintenance", 
        "utilities", "management_fees"
    ]
    
    for component in top_level_components:
        if component not in normalized and component in financials:
            normalized[component] = financials[component]
    
    # Extract Other Income components that might be at the top level
    top_level_components = [
        "parking", "laundry", "late_fees", "pet_fees", "application_fees"
    ]
    
    for component in top_level_components:
        if component not in normalized and component in financials:
            normalized[component] = financials[component]
    
    # Calculate missing OpEx components with sensible defaults
    if "opex" in normalized:
        # If we have total OpEx but missing components, set defaults
        for component in ["property_taxes", "insurance", "repairs_and_maintenance", "utilities", "management_fees"]:
            if component not in normalized:
                normalized[component] = 0
    
    # Calculate missing Other Income components with sensible defaults
    if "other_income" in normalized and normalized["other_income"] > 0:
        # If we have total Other Income but missing components, set defaults
        for component in ["parking", "laundry", "late_fees", "pet_fees", "application_fees"]:
            if component not in normalized:
                normalized[component] = 0
    
    return normalized

def format_comparison_results(comparison_results: Dict[str, Any], currency_format: bool = True) -> Dict[str, Any]:
    """
    Format comparison results for display or export
    
    Args:
        comparison_results: Raw comparison results
        currency_format: Whether to format values as currency strings
        
    Returns:
        Formatted comparison results
    """
    formatted = {}
    
    # Copy metadata
    if "current_data" in comparison_results:
        formatted["current_metadata"] = comparison_results["current_data"].get("metadata", {})
    
    if "compare_data" in comparison_results:
        formatted["compare_metadata"] = comparison_results["compare_data"].get("metadata", {})
    
    # Format each comparison metric
    for key, value in comparison_results.items():
        # Skip metadata
        if key in ["current_data", "compare_data"]:
            continue
        
        # Format numeric values if currency_format is True
        if currency_format and isinstance(value, (int, float)):
            formatted[key] = "${:,.2f}".format(value)
        else:
            formatted[key] = value
    
    return formatted 