"""
Validation and Output Formatting Module for Real Estate NOI Analyzer
Validates extracted financial data and formats it according to the required JSON structure
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('validation_formatter')

class ValidationFormatter:
    """
    Class for validating and formatting extracted financial data
    """
    
    def __init__(self):
        """Initialize the validation and formatting module"""
        # Define expected fields and their types
        self.expected_fields = {
            'document_type': str,
            'period': str,
            'rental_income': (float, int),
            'laundry_income': (float, int),
            'parking_income': (float, int),
            'other_revenue': (float, int),
            'total_revenue': (float, int),
            'repairs_maintenance': (float, int),
            'utilities': (float, int),
            'property_management_fees': (float, int),
            'property_taxes': (float, int),
            'insurance': (float, int),
            'admin_office_costs': (float, int),
            'marketing_advertising': (float, int),
            'total_expenses': (float, int),
            'net_operating_income': (float, int)
        }
        
        # Define validation rules
        self.validation_rules = [
            self._validate_document_type,
            self._validate_period,
            self._validate_numeric_fields,
            self._validate_total_revenue,
            self._validate_total_expenses,
            self._validate_noi
        ]
    
    def validate_and_format(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate and format the extracted financial data
        
        Args:
            data: Extracted financial data
            
        Returns:
            Tuple containing:
                - Formatted data in the required JSON structure
                - List of validation warnings
        """
        logger.info("Validating and formatting extracted financial data")
        
        # Initialize warnings list
        warnings = []
        
        # Make a copy of the data to avoid modifying the original
        formatted_data = data.copy()
        
        # Ensure all expected fields are present
        for field in self.expected_fields:
            if field not in formatted_data:
                formatted_data[field] = None
                warnings.append(f"Missing field: {field}")
        
        # Convert numeric fields to appropriate types
        for field, field_type in self.expected_fields.items():
            if field in formatted_data and formatted_data[field] is not None:
                if field_type in [(float, int), float, int]:
                    try:
                        if isinstance(formatted_data[field], str):
                            # Remove commas and other non-numeric characters
                            clean_value = formatted_data[field].replace(',', '').replace('$', '')
                            formatted_data[field] = float(clean_value)
                    except (ValueError, TypeError):
                        warnings.append(f"Invalid numeric value for {field}: {formatted_data[field]}")
                        formatted_data[field] = None
        
        # Run all validation rules
        for rule in self.validation_rules:
            rule_warnings = rule(formatted_data)
            warnings.extend(rule_warnings)
        
        # Format the data according to the required structure
        final_output = self._format_output(formatted_data)
        
        return final_output, warnings
    
    def _validate_document_type(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate document_type field
        
        Args:
            data: Extracted financial data
            
        Returns:
            List of validation warnings
        """
        warnings = []
        valid_types = ["Actual", "Budget", "Prior Year Actual", "Unknown"]
        
        if data.get('document_type') not in valid_types:
            warnings.append(f"Invalid document_type: {data.get('document_type')}. Expected one of {valid_types}")
        
        return warnings
    
    def _validate_period(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate period field
        
        Args:
            data: Extracted financial data
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        if not data.get('period'):
            warnings.append("Missing period")
        
        return warnings
    
    def _validate_numeric_fields(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate numeric fields
        
        Args:
            data: Extracted financial data
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        for field, field_type in self.expected_fields.items():
            if field_type in [(float, int), float, int]:
                if data.get(field) is not None and not isinstance(data[field], (float, int)):
                    warnings.append(f"Field {field} should be numeric, got {type(data[field]).__name__}")
        
        return warnings
    
    def _validate_total_revenue(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate total_revenue field
        
        Args:
            data: Extracted financial data
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        # Check if we have all the revenue components and total_revenue
        revenue_fields = ['rental_income', 'laundry_income', 'parking_income', 'other_revenue']
        if all(data.get(field) is not None for field in revenue_fields) and data.get('total_revenue') is not None:
            # Calculate expected total revenue
            expected_total = sum(data.get(field, 0) or 0 for field in revenue_fields)
            actual_total = data.get('total_revenue', 0) or 0
            
            # Allow for small rounding differences
            if abs(expected_total - actual_total) > 1:
                warnings.append(f"Total revenue ({actual_total}) doesn't match sum of revenue items ({expected_total})")
                # Correct the total revenue
                data['total_revenue'] = expected_total
        
        return warnings
    
    def _validate_total_expenses(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate total_expenses field
        
        Args:
            data: Extracted financial data
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        # Check if we have all the expense components and total_expenses
        expense_fields = [
            'repairs_maintenance', 'utilities', 'property_management_fees',
            'property_taxes', 'insurance', 'admin_office_costs', 'marketing_advertising'
        ]
        
        if all(data.get(field) is not None for field in expense_fields) and data.get('total_expenses') is not None:
            # Calculate expected total expenses
            expected_total = sum(data.get(field, 0) or 0 for field in expense_fields)
            actual_total = data.get('total_expenses', 0) or 0
            
            # Allow for small rounding differences
            if abs(expected_total - actual_total) > 1:
                warnings.append(f"Total expenses ({actual_total}) doesn't match sum of expense items ({expected_total})")
                # Correct the total expenses
                data['total_expenses'] = expected_total
        
        return warnings
    
    def _validate_noi(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate net_operating_income field
        
        Args:
            data: Extracted financial data
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        # Check if we have total_revenue, total_expenses, and net_operating_income
        if data.get('total_revenue') is not None and data.get('total_expenses') is not None and data.get('net_operating_income') is not None:
            # Calculate expected NOI
            total_revenue = data.get('total_revenue', 0) or 0
            total_expenses = data.get('total_expenses', 0) or 0
            expected_noi = total_revenue - total_expenses
            actual_noi = data.get('net_operating_income', 0) or 0
            
            # Allow for small rounding differences
            if abs(expected_noi - actual_noi) > 1:
                warnings.append(f"NOI ({actual_noi}) doesn't match total_revenue - total_expenses ({expected_noi})")
                # Correct the NOI
                data['net_operating_income'] = expected_noi
        
        return warnings
    
    def _format_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the data according to the required JSON structure
        
        Args:
            data: Validated financial data
            
        Returns:
            Formatted data in the required JSON structure
        """
        # Create the output structure
        output = {
            "document_type": data.get('document_type', 'Unknown'),
            "period": data.get('period', ''),
            "financials": {
                "rental_income": data.get('rental_income'),
                "laundry_income": data.get('laundry_income'),
                "parking_income": data.get('parking_income'),
                "other_revenue": data.get('other_revenue'),
                "total_revenue": data.get('total_revenue'),
                "repairs_maintenance": data.get('repairs_maintenance'),
                "utilities": data.get('utilities'),
                "property_management_fees": data.get('property_management_fees'),
                "property_taxes": data.get('property_taxes'),
                "insurance": data.get('insurance'),
                "admin_office_costs": data.get('admin_office_costs'),
                "marketing_advertising": data.get('marketing_advertising'),
                "total_expenses": data.get('total_expenses'),
                "net_operating_income": data.get('net_operating_income')
            }
        }
        
        # Remove None values from financials
        output["financials"] = {k: v for k, v in output["financials"].items() if v is not None}
        
        return output


def validate_and_format_data(data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Convenience function to validate and format extracted financial data
    
    Args:
        data: Extracted financial data
        
    Returns:
        Tuple containing:
            - Formatted data in the required JSON structure
            - List of validation warnings
    """
    validator = ValidationFormatter()
    return validator.validate_and_format(data)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validation_formatter.py <json_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        formatted_data, warnings = validate_and_format_data(data)
        
        print("Formatted Data:")
        print(json.dumps(formatted_data, indent=2))
        
        if warnings:
            print("\nValidation Warnings:")
            for warning in warnings:
                print(f"- {warning}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
