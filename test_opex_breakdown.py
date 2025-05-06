import unittest
import json
from noi_comparison import calculate_noi_comparisons, _normalize_financials_data
from main_backend import format_for_noi_analyzer
from typing import Dict, Any

class TestOpExBreakdown(unittest.TestCase):
    """Test the OpEx breakdown functionality"""
    
    def test_normalize_financials_data(self):
        """Test normalizing financial data with OpEx components"""
        # Test nested structure
        nested_data = {
            "financials": {
                "gross_potential_rent": 100000,
                "vacancy_loss": 5000,
                "opex": 40000,
                "noi": 55000,
                "operating_expenses": {
                    "property_taxes": 15000,
                    "insurance": 5000,
                    "repairs_maintenance": 10000,
                    "utilities": 8000,
                    "management_fees": 2000,
                    "total_operating_expenses": 40000
                }
            }
        }
        
        normalized = _normalize_financials_data(nested_data)
        self.assertEqual(normalized["property_taxes"], 15000)
        self.assertEqual(normalized["insurance"], 5000)
        self.assertEqual(normalized["repairs_and_maintenance"], 10000)
        self.assertEqual(normalized["utilities"], 8000)
        self.assertEqual(normalized["management_fees"], 2000)
        self.assertEqual(normalized["opex"], 40000)
        
        # Test flat structure
        flat_data = {
            "gross_potential_rent": 100000,
            "vacancy_loss": 5000,
            "operating_expenses_total": 40000,
            "property_taxes": 15000,
            "insurance": 5000,
            "repairs_and_maintenance": 10000,
            "utilities": 8000,
            "management_fees": 2000,
            "noi": 55000
        }
        
        normalized = _normalize_financials_data(flat_data)
        self.assertEqual(normalized["property_taxes"], 15000)
        self.assertEqual(normalized["insurance"], 5000)
        self.assertEqual(normalized["repairs_and_maintenance"], 10000)
        self.assertEqual(normalized["utilities"], 8000)
        self.assertEqual(normalized["management_fees"], 2000)
        self.assertEqual(normalized["opex"], 40000)
    
    def test_calculate_noi_comparisons_with_opex_breakdown(self):
        """Test calculating NOI comparisons with OpEx breakdown"""
        current_data = {
            "financials": {
                "gross_potential_rent": 100000,
                "vacancy_loss": 5000,
                "effective_gross_income": 95000,
                "opex": 40000,
                "property_taxes": 15000,
                "insurance": 5000,
                "repairs_and_maintenance": 10000,
                "utilities": 8000,
                "management_fees": 2000,
                "noi": 55000
            }
        }
        
        compare_data = {
            "financials": {
                "gross_potential_rent": 95000,
                "vacancy_loss": 4750,
                "effective_gross_income": 90250,
                "opex": 38000,
                "property_taxes": 14000,
                "insurance": 4800,
                "repairs_and_maintenance": 9500,
                "utilities": 7700,
                "management_fees": 2000,
                "noi": 52250
            }
        }
        
        result = calculate_noi_comparisons(current_data, compare_data)
        
        # Check that all OpEx components are included in the comparison
        self.assertIn("property_taxes_current", result)
        self.assertIn("property_taxes_compare", result)
        self.assertIn("property_taxes_change", result)
        self.assertIn("property_taxes_percent_change", result)
        
        self.assertIn("insurance_current", result)
        self.assertIn("repairs_and_maintenance_current", result)
        self.assertIn("utilities_current", result)
        self.assertIn("management_fees_current", result)
        
        # Check comparison calculations
        self.assertEqual(result["property_taxes_current"], 15000)
        self.assertEqual(result["property_taxes_compare"], 14000)
        self.assertEqual(result["property_taxes_change"], 1000)
        self.assertAlmostEqual(result["property_taxes_percent_change"], 7.14, places=2)
    
    def test_format_for_noi_analyzer(self):
        """Test formatting extraction results with OpEx breakdown"""
        # Test with nested operating expenses
        extraction_result = {
            "property_id": "123",
            "period": "Dec 2023",
            "gross_potential_rent": 100000,
            "vacancy_loss": 5000,
            "effective_gross_income": 95000,
            "net_operating_income": 55000,
            "operating_expenses": {
                "property_taxes": 15000,
                "insurance": 5000,
                "repairs_maintenance": 10000,
                "utilities": 8000,
                "management_fees": 2000,
                "total_operating_expenses": 40000
            },
            "metadata": {
                "filename": "test.pdf"
            }
        }
        
        result = format_for_noi_analyzer(extraction_result)
        
        # Check that all OpEx components are included in the result
        self.assertIn("property_taxes", result["financials"])
        self.assertIn("insurance", result["financials"])
        self.assertIn("repairs_and_maintenance", result["financials"])
        self.assertIn("utilities", result["financials"])
        self.assertIn("management_fees", result["financials"])
        
        self.assertEqual(result["financials"]["property_taxes"], 15000)
        self.assertEqual(result["financials"]["insurance"], 5000)
        self.assertEqual(result["financials"]["repairs_and_maintenance"], 10000)
        self.assertEqual(result["financials"]["utilities"], 8000)
        self.assertEqual(result["financials"]["management_fees"], 2000)
        self.assertEqual(result["financials"]["total_expenses"], 40000)
        
if __name__ == "__main__":
    unittest.main() 