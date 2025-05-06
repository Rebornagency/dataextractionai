import unittest
import json
from noi_comparison import calculate_noi_comparisons, _normalize_financials_data
from main_backend import format_for_noi_analyzer
from typing import Dict, Any

class TestOtherIncomeBreakdown(unittest.TestCase):
    """Test the Other Income breakdown functionality"""
    
    def test_normalize_financials_data(self):
        """Test normalizing financial data with Other Income components"""
        # Test nested structure
        nested_data = {
            "financials": {
                "gross_potential_rent": 100000,
                "vacancy_loss": 5000,
                "other_income": {
                    "parking": 2100,
                    "laundry": 1500,
                    "late_fees": 900,
                    "pet_fees": 400,
                    "application_fees": 300,
                    "total": 5200
                },
                "opex": 40000,
                "noi": 60200
            }
        }
        
        normalized = _normalize_financials_data(nested_data)
        self.assertEqual(normalized["other_income"], 5200)
        self.assertEqual(normalized["parking"], 2100)
        self.assertEqual(normalized["laundry"], 1500)
        self.assertEqual(normalized["late_fees"], 900)
        self.assertEqual(normalized["pet_fees"], 400)
        self.assertEqual(normalized["application_fees"], 300)
        
        # Test flat structure
        flat_data = {
            "gross_potential_rent": 100000,
            "vacancy_loss": 5000,
            "other_income": 5200,
            "parking": 2100,
            "laundry": 1500,
            "late_fees": 900,
            "pet_fees": 400,
            "application_fees": 300,
            "operating_expenses_total": 40000,
            "noi": 60200
        }
        
        normalized = _normalize_financials_data(flat_data)
        self.assertEqual(normalized["other_income"], 5200)
        self.assertEqual(normalized["parking"], 2100)
        self.assertEqual(normalized["laundry"], 1500)
        self.assertEqual(normalized["late_fees"], 900)
        self.assertEqual(normalized["pet_fees"], 400)
        self.assertEqual(normalized["application_fees"], 300)
    
    def test_calculate_noi_comparisons_with_other_income_breakdown(self):
        """Test calculating NOI comparisons with Other Income breakdown"""
        current_data = {
            "financials": {
                "gross_potential_rent": 100000,
                "vacancy_loss": 5000,
                "effective_gross_income": 100200,
                "other_income": 5200,
                "parking": 2100,
                "laundry": 1500,
                "late_fees": 900,
                "pet_fees": 400,
                "application_fees": 300,
                "opex": 40000,
                "noi": 60200
            }
        }
        
        compare_data = {
            "financials": {
                "gross_potential_rent": 95000,
                "vacancy_loss": 4750,
                "effective_gross_income": 95200,
                "other_income": 4950,
                "parking": 2000,
                "laundry": 1450,
                "late_fees": 850,
                "pet_fees": 350,
                "application_fees": 300,
                "opex": 38000,
                "noi": 57200
            }
        }
        
        result = calculate_noi_comparisons(current_data, compare_data)
        
        # Check that all Other Income components are included in the comparison
        self.assertIn("parking_current", result)
        self.assertIn("parking_compare", result)
        self.assertIn("parking_change", result)
        self.assertIn("parking_percent_change", result)
        
        self.assertIn("laundry_current", result)
        self.assertIn("late_fees_current", result)
        self.assertIn("pet_fees_current", result)
        self.assertIn("application_fees_current", result)
        
        # Check comparison calculations
        self.assertEqual(result["parking_current"], 2100)
        self.assertEqual(result["parking_compare"], 2000)
        self.assertEqual(result["parking_change"], 100)
        self.assertEqual(result["parking_percent_change"], 5.0)
    
    def test_format_for_noi_analyzer(self):
        """Test formatting extraction results with Other Income breakdown"""
        # Test with nested other income
        extraction_result = {
            "property_id": "123",
            "period": "Dec 2023",
            "gross_potential_rent": 100000,
            "vacancy_loss": 5000,
            "effective_gross_income": 100200,
            "net_operating_income": 60200,
            "other_income": {
                "parking": 2100,
                "laundry": 1500,
                "late_fees": 900,
                "pet_fees": 400,
                "application_fees": 300,
                "total": 5200
            },
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
        
        # Check that all Other Income components are included in the result
        self.assertIn("other_income", result["financials"])
        self.assertIn("parking", result["financials"])
        self.assertIn("laundry", result["financials"])
        self.assertIn("late_fees", result["financials"])
        self.assertIn("pet_fees", result["financials"])
        self.assertIn("application_fees", result["financials"])
        
        self.assertEqual(result["financials"]["other_income"], 5200)
        self.assertEqual(result["financials"]["parking"], 2100)
        self.assertEqual(result["financials"]["laundry"], 1500)
        self.assertEqual(result["financials"]["late_fees"], 900)
        self.assertEqual(result["financials"]["pet_fees"], 400)
        self.assertEqual(result["financials"]["application_fees"], 300)
        
if __name__ == "__main__":
    unittest.main() 