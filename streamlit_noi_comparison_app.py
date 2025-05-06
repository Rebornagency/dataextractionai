import streamlit as st
import pandas as pd
import json
from typing import Dict, Any, Optional
import os
import sys
from datetime import datetime

# Add the current directory to the path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our comparison functionality
from noi_comparison import calculate_noi_comparisons, format_comparison_results

def display_comparison_tab(current_data: Dict[str, Any], compare_data: Optional[Dict[str, Any]] = None, 
                          show_zero_values: bool = True):
    """
    Display the NOI comparison tab with OpEx breakdown
    
    Args:
        current_data: Current period financial data
        compare_data: Prior period financial data for comparison (optional)
        show_zero_values: Whether to include zero values in the display
    """
    # Calculate comparison metrics
    comparison_results = calculate_noi_comparisons(current_data, compare_data, show_zero_values)
    
    # Display header with property info
    st.header("NOI Comparison Analysis")
    
    # Display property information
    property_id = current_data.get("property_id", "Unknown")
    current_period = current_data.get("period", "Current")
    compare_period = compare_data.get("period", "Prior") if compare_data else None
    
    st.subheader(f"Property ID: {property_id}")
    st.write(f"Current Period: {current_period}")
    if compare_period:
        st.write(f"Compare Period: {compare_period}")
    
    # Display high-level KPIs
    st.subheader("Key Financial Metrics")
    
    # Create columns for the KPIs
    cols = st.columns(3)
    
    # Define the 6 primary KPIs
    kpis = [
        {"name": "Gross Potential Rent", "key": "gross_potential_rent"},
        {"name": "Vacancy Loss", "key": "vacancy_loss"},
        {"name": "Effective Gross Income", "key": "effective_gross_income"},
        {"name": "Operating Expenses", "key": "opex"},
        {"name": "Net Operating Income", "key": "noi"},
        {"name": "Expense Ratio", "key": "expense_ratio", "custom": True}
    ]
    
    # Calculate expense ratio
    if "opex_current" in comparison_results and "effective_gross_income_current" in comparison_results:
        if comparison_results["effective_gross_income_current"] != 0:
            expense_ratio_current = comparison_results["opex_current"] / comparison_results["effective_gross_income_current"] * 100
        else:
            expense_ratio_current = 0
        
        comparison_results["expense_ratio_current"] = expense_ratio_current
        
        if compare_data and "opex_compare" in comparison_results and "effective_gross_income_compare" in comparison_results:
            if comparison_results["effective_gross_income_compare"] != 0:
                expense_ratio_compare = comparison_results["opex_compare"] / comparison_results["effective_gross_income_compare"] * 100
            else:
                expense_ratio_compare = 0
            
            comparison_results["expense_ratio_compare"] = expense_ratio_compare
            comparison_results["expense_ratio_change"] = expense_ratio_current - expense_ratio_compare
            if expense_ratio_compare != 0:
                comparison_results["expense_ratio_percent_change"] = (expense_ratio_current - expense_ratio_compare) / expense_ratio_compare * 100
            else:
                comparison_results["expense_ratio_percent_change"] = 0
    
    # Display KPIs in columns
    for i, kpi in enumerate(kpis):
        col = cols[i % 3]
        with col:
            st.metric(
                label=kpi["name"],
                value=f"${comparison_results.get(f'{kpi['key']}_current', 0):,.2f}" if not kpi.get("custom") else f"{comparison_results.get(f'{kpi['key']}_current', 0):.1f}%",
                delta=f"${comparison_results.get(f'{kpi['key']}_change', 0):,.2f}" if compare_data and not kpi.get("custom") else f"{comparison_results.get(f'{kpi['key']}_change', 0):.1f}%" if compare_data and kpi.get("custom") else None
            )
    
    # Create Other Income Breakdown expander
    with st.expander("Other Income Breakdown", expanded=True):
        # Create DataFrame for Other Income breakdown
        income_breakdown_data = {}
        income_metrics = ["parking", "laundry", "late_fees", "pet_fees", "application_fees"]
        
        for metric in income_metrics:
            current_value = comparison_results.get(f"{metric}_current", 0)
            compare_value = comparison_results.get(f"{metric}_compare", None)
            change = comparison_results.get(f"{metric}_change", None)
            percent_change = comparison_results.get(f"{metric}_percent_change", None)
            
            if current_value == 0 and not show_zero_values:
                continue
                
            # Format metric name for display
            display_name = " ".join([word.capitalize() for word in metric.split("_")])
            
            # Add data row
            income_breakdown_data[display_name] = {
                "Current": current_value,
                "Prior": compare_value if compare_data else None,
                "Change": change if compare_data else None,
                "% Change": f"{percent_change:.1f}%" if compare_data and percent_change is not None else None
            }
        
        # Create DataFrame
        income_df = pd.DataFrame.from_dict(income_breakdown_data, orient="index")
        
        # Format currency columns
        for col in ["Current", "Prior", "Change"]:
            if col in income_df.columns:
                income_df[col] = income_df[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
        
        # Display the DataFrame
        st.dataframe(income_df)
        
        # Add a pie chart visualization
        if any(comparison_results.get(f"{metric}_current", 0) for metric in income_metrics):
            st.subheader("Other Income Distribution")
            
            # Create pie chart data
            income_pie_data = {
                "values": [comparison_results.get(f"{metric}_current", 0) for metric in income_metrics if comparison_results.get(f"{metric}_current", 0) > 0],
                "labels": [" ".join([word.capitalize() for word in metric.split("_")]) for metric in income_metrics if comparison_results.get(f"{metric}_current", 0) > 0]
            }
            
            # Only show pie chart if we have values
            if income_pie_data["values"]:
                fig = {
                    "data": [{
                        "values": income_pie_data["values"],
                        "labels": income_pie_data["labels"],
                        "type": "pie",
                        "hole": 0.4
                    }],
                    "layout": {"height": 300}
                }
                st.plotly_chart(fig)
    
    # Create OpEx Breakdown expander
    with st.expander("OpEx Breakdown", expanded=True):
        # Create DataFrame for OpEx breakdown
        breakdown_data = {}
        metrics = ["property_taxes", "insurance", "repairs_and_maintenance", "utilities", "management_fees"]
        
        for metric in metrics:
            current_value = comparison_results.get(f"{metric}_current", 0)
            compare_value = comparison_results.get(f"{metric}_compare", None)
            change = comparison_results.get(f"{metric}_change", None)
            percent_change = comparison_results.get(f"{metric}_percent_change", None)
            
            if current_value == 0 and not show_zero_values:
                continue
                
            # Format metric name for display
            display_name = " ".join([word.capitalize() for word in metric.split("_")])
            
            # Add data row
            breakdown_data[display_name] = {
                "Current": current_value,
                "Prior": compare_value if compare_data else None,
                "Change": change if compare_data else None,
                "% Change": f"{percent_change:.1f}%" if compare_data and percent_change is not None else None
            }
        
        # Create DataFrame
        df = pd.DataFrame.from_dict(breakdown_data, orient="index")
        
        # Format currency columns
        for col in ["Current", "Prior", "Change"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
        
        # Display the DataFrame
        st.dataframe(df)
        
        # Add a pie chart visualization
        if any(comparison_results.get(f"{metric}_current", 0) for metric in metrics):
            st.subheader("OpEx Distribution")
            
            # Create pie chart data
            pie_data = {
                "values": [comparison_results.get(f"{metric}_current", 0) for metric in metrics if comparison_results.get(f"{metric}_current", 0) > 0],
                "labels": [" ".join([word.capitalize() for word in metric.split("_")]) for metric in metrics if comparison_results.get(f"{metric}_current", 0) > 0]
            }
            
            # Only show pie chart if we have values
            if pie_data["values"]:
                fig = {
                    "data": [{
                        "values": pie_data["values"],
                        "labels": pie_data["labels"],
                        "type": "pie",
                        "hole": 0.4
                    }],
                    "layout": {"height": 300}
                }
                st.plotly_chart(fig)

def main():
    st.set_page_config(
        page_title="NOI Analyzer - Comparison Demo",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("NOI Analyzer - Detailed Breakdown Demo")
    
    # Create sample data
    sample_current_data = {
        "property_id": "123 Main Street",
        "period": "Dec 2023",
        "financials": {
            "gross_potential_rent": 750000,
            "vacancy_loss": 37500,
            "concessions": 15000,
            "bad_debt": 7500,
            "other_income": 45000,
            "parking": 18000,
            "laundry": 12000, 
            "late_fees": 6000,
            "pet_fees": 5000,
            "application_fees": 4000,
            "effective_gross_income": 735000,
            "opex": 286000,
            "property_taxes": 95000,
            "insurance": 35000,
            "repairs_and_maintenance": 75000,
            "utilities": 55000,
            "management_fees": 26000,
            "noi": 449000,
        }
    }
    
    sample_compare_data = {
        "property_id": "123 Main Street",
        "period": "Dec 2022",
        "financials": {
            "gross_potential_rent": 725000,
            "vacancy_loss": 43500,
            "concessions": 14500,
            "bad_debt": 7250,
            "other_income": 42000,
            "parking": 16800,
            "laundry": 11400,
            "late_fees": 5600,
            "pet_fees": 4500,
            "application_fees": 3700,
            "effective_gross_income": 701750,
            "opex": 272450,
            "property_taxes": 91000,
            "insurance": 33500,
            "repairs_and_maintenance": 71000,
            "utilities": 52500,
            "management_fees": 24450,
            "noi": 429300,
        }
    }
    
    # Create sidebar
    st.sidebar.header("Controls")
    
    # Add toggle for show zero values
    show_zero_values = st.sidebar.checkbox("Show Zero Values", value=True)
    
    # Add option for data manipulation
    st.sidebar.subheader("Operating Expenses")
    property_taxes = st.sidebar.slider("Property Taxes", 0, 150000, int(sample_current_data["financials"]["property_taxes"]), 1000)
    insurance = st.sidebar.slider("Insurance", 0, 100000, int(sample_current_data["financials"]["insurance"]), 1000)
    repairs = st.sidebar.slider("Repairs & Maintenance", 0, 100000, int(sample_current_data["financials"]["repairs_and_maintenance"]), 1000)
    utilities = st.sidebar.slider("Utilities", 0, 100000, int(sample_current_data["financials"]["utilities"]), 1000)
    management_fees = st.sidebar.slider("Management Fees", 0, 50000, int(sample_current_data["financials"]["management_fees"]), 1000)
    
    # Add option for Other Income manipulation
    st.sidebar.subheader("Other Income")
    parking = st.sidebar.slider("Parking", 0, 30000, int(sample_current_data["financials"]["parking"]), 500)
    laundry = st.sidebar.slider("Laundry", 0, 20000, int(sample_current_data["financials"]["laundry"]), 500)
    late_fees = st.sidebar.slider("Late Fees", 0, 10000, int(sample_current_data["financials"]["late_fees"]), 250)
    pet_fees = st.sidebar.slider("Pet Fees", 0, 10000, int(sample_current_data["financials"]["pet_fees"]), 250)
    app_fees = st.sidebar.slider("Application Fees", 0, 8000, int(sample_current_data["financials"]["application_fees"]), 250)
    
    # Update current data based on sliders
    sample_current_data["financials"]["property_taxes"] = property_taxes
    sample_current_data["financials"]["insurance"] = insurance
    sample_current_data["financials"]["repairs_and_maintenance"] = repairs
    sample_current_data["financials"]["utilities"] = utilities
    sample_current_data["financials"]["management_fees"] = management_fees
    sample_current_data["financials"]["parking"] = parking
    sample_current_data["financials"]["laundry"] = laundry
    sample_current_data["financials"]["late_fees"] = late_fees
    sample_current_data["financials"]["pet_fees"] = pet_fees
    sample_current_data["financials"]["application_fees"] = app_fees
    
    # Recalculate opex and other_income totals
    total_opex = property_taxes + insurance + repairs + utilities + management_fees
    sample_current_data["financials"]["opex"] = total_opex
    
    total_other_income = parking + laundry + late_fees + pet_fees + app_fees
    sample_current_data["financials"]["other_income"] = total_other_income
    
    # Recalculate effective gross income (EGI) and NOI
    gpr = sample_current_data["financials"]["gross_potential_rent"]
    vacancy = sample_current_data["financials"]["vacancy_loss"]
    concessions = sample_current_data["financials"]["concessions"]
    bad_debt = sample_current_data["financials"]["bad_debt"]
    
    egi = gpr - vacancy - concessions - bad_debt + total_other_income
    sample_current_data["financials"]["effective_gross_income"] = egi
    sample_current_data["financials"]["noi"] = egi - total_opex
    
    # Add compare data toggle
    compare_enabled = st.sidebar.checkbox("Enable Comparison Data", value=True)
    
    # Display comparison tab
    if compare_enabled:
        display_comparison_tab(sample_current_data, sample_compare_data, show_zero_values)
    else:
        display_comparison_tab(sample_current_data, None, show_zero_values)
    
    # Add export options
    st.sidebar.header("Export Options")
    
    export_format = st.sidebar.selectbox("Export Format", ["Excel", "CSV", "JSON"])
    
    if st.sidebar.button("Export Data"):
        comparison_results = calculate_noi_comparisons(sample_current_data, sample_compare_data if compare_enabled else None, show_zero_values)
        
        if export_format == "JSON":
            st.sidebar.download_button(
                label="Download JSON",
                data=json.dumps(comparison_results, indent=2),
                file_name=f"noi_comparison_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        elif export_format == "Excel":
            st.sidebar.info("Excel export functionality would be implemented here")
        elif export_format == "CSV":
            # Create DataFrame from comparison results
            df_data = {}
            
            for key, value in comparison_results.items():
                if key not in ["current_data", "compare_data"] and isinstance(value, (int, float)):
                    metric_parts = key.split("_")
                    if len(metric_parts) > 1:
                        metric = "_".join(metric_parts[:-1])
                        stat = metric_parts[-1]
                        
                        if metric not in df_data:
                            df_data[metric] = {}
                        
                        df_data[metric][stat] = value
            
            # Convert to DataFrame
            export_df = pd.DataFrame.from_dict(df_data, orient="index")
            
            # Download button
            st.sidebar.download_button(
                label="Download CSV",
                data=export_df.to_csv(index=True),
                file_name=f"noi_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # Add disclaimer
    st.sidebar.markdown("---")
    st.sidebar.info("This is a demo of the NOI Analyzer with detailed breakdowns. In a production environment, this would be integrated with the real data extraction pipeline.")

if __name__ == "__main__":
    main() 