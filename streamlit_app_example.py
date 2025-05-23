import streamlit as st
import json
from typing import Dict, Any, List
import logging
import traceback
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('streamlit_app')

# Import the extract_noi_data function
from api_server import extract_noi_data

def display_extraction_results(results: Dict[str, Any]):
    """
    Display extraction results with validation warnings
    
    Args:
        results: Extraction results
    """
    if not results:
        st.error("No results to display")
        return
        
    # Display each result
    for filename, result in results.items():
        st.subheader(f"Results for {filename}")
        
        # Check if there's an error
        if "error" in result:
            st.error(f"Error: {result['error']}")
            st.info("Please try again or contact support if the problem persists.")
            continue
            
        # Check for validation issues
        if "validation_issues" in result:
            for issue in result["validation_issues"]:
                st.warning(f"Validation issue: {issue}")
                
            # Show raw extraction lines in an expander for debugging
            with st.expander("Show GPT Output"):
                if "audit_lines" in result:
                    st.write("### Raw Extraction Lines")
                    for line in result["audit_lines"]:
                        st.text(line)
                else:
                    st.info("No raw extraction lines available")
        
        # Display financial data
        st.write("### Financial Data")
        
        # Format the display of financial data
        financial_data = {k: v for k, v in result.items() 
                        if k not in ["metadata", "audit_lines", "validation_issues"]}
        
        # Pretty display for each section
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Income:")
            st.write(f"- Gross Potential Rent: ${result.get('gross_potential_rent', 0):,.2f}")
            st.write(f"- Vacancy Loss: ${result.get('vacancy_loss', 0):,.2f}")
            st.write(f"- Other Income: ${result.get('other_income', {}).get('total', 0):,.2f}")
            st.write(f"- Effective Gross Income: ${result.get('effective_gross_income', 0):,.2f}")
            
        with col2:
            st.write("Expenses:")
            st.write(f"- Operating Expenses: ${result.get('operating_expenses', {}).get('total_operating_expenses', 0):,.2f}")
            st.write(f"- Reserves: ${result.get('reserves', {}).get('total_reserves', 0):,.2f}")
            st.write(f"- Net Operating Income: ${result.get('net_operating_income', 0):,.2f}")
        
        # Add Other Income Breakdown section
        with st.expander("Other Income Breakdown"):
            # Get financials data, either from nested structure or top level
            financials = result.get('financials', result)
            
            # Get other income data, either from nested structure or top level
            other_income_data = financials.get('other_income', {})
            if not isinstance(other_income_data, dict):
                other_income_data = {}
            
            # Create breakdown dictionary
            income_breakdown = {
                "Parking": financials.get("parking", other_income_data.get("parking", 0)),
                "Laundry": financials.get("laundry", other_income_data.get("laundry", 0)),
                "Late Fees": financials.get("late_fees", other_income_data.get("late_fees", 0)),
                "Pet Fees": financials.get("pet_fees", other_income_data.get("pet_fees", 0)),
                "Application Fees": financials.get("application_fees", other_income_data.get("application_fees", 0))
            }
            
            # Convert to DataFrame
            income_df = pd.DataFrame.from_dict(income_breakdown, orient="index", columns=["Value"])
            
            # Format as currency
            st.dataframe(income_df.style.format("${:,.2f}"))
            
            # Add a pie chart visualization
            if any(income_breakdown.values()):
                st.write("### Other Income Distribution")
                fig = {
                    "data": [{
                        "values": list(income_breakdown.values()),
                        "labels": list(income_breakdown.keys()),
                        "type": "pie",
                        "hole": 0.4
                    }],
                    "layout": {"height": 300}
                }
                st.plotly_chart(fig)
        
        # Add OpEx Breakdown section
        with st.expander("OpEx Breakdown"):
            # Build a DataFrame of each expense line and its value
            # Get financials data, either from nested structure or top level
            financials = result.get('financials', result)
            
            # Create breakdown dictionary
            breakdown = {
                "Property Taxes": financials.get("property_taxes", 0),
                "Insurance": financials.get("insurance", 0),
                "Repairs & Maintenance": financials.get("repairs_and_maintenance", 0),
                "Utilities": financials.get("utilities", 0),
                "Management Fees": financials.get("management_fees", 0)
            }
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(breakdown, orient="index", columns=["Value"])
            
            # Format as currency
            st.dataframe(df.style.format("${:,.2f}"))
            
            # Add a pie chart visualization
            if any(breakdown.values()):
                st.write("### OpEx Distribution")
                fig = {
                    "data": [{
                        "values": list(breakdown.values()),
                        "labels": list(breakdown.keys()),
                        "type": "pie",
                        "hole": 0.4
                    }],
                    "layout": {"height": 300}
                }
                st.plotly_chart(fig)
        
        # Display metadata
        st.write("### Metadata")
        st.json(result.get("metadata", {}))
        
def handle_extraction_with_error_handling(files):
    """
    Handle the extraction process with proper error handling
    
    Args:
        files: List of uploaded files
        
    Returns:
        Extraction results dictionary
    """
    try:
        with st.spinner("Processing files..."):
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Placeholder for results
            results = {}
            
            for i, file in enumerate(files):
                # Update progress
                progress = (i + 1) / len(files)
                progress_bar.progress(progress)
                status_text.text(f"Processing {file.name}...")
                
                try:
                    # Call extraction function
                    result = extract_noi_data([file])
                    results[file.name] = result.get(file.name, {"error": "Processing failed"})
                except Exception as e:
                    logger.error(f"Error processing {file.name}: {str(e)}", exc_info=True)
                    results[file.name] = {
                        "error": f"Failed to process file: {str(e)}",
                        "status": "error"
                    }
                
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            return results
            
    except Exception as e:
        logger.error(f"Extraction process failed: {str(e)}", exc_info=True)
        st.error("Oops! We hit a problem during extraction. Please try again or contact support.")
        st.error(f"Error details: {str(e)}")
        return {}
        
def main():
    st.title("NOI Analyzer - Data Extraction")
    
    # File uploader
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf", "xlsx", "xls", "csv"], accept_multiple_files=True)
    
    if uploaded_files:
        # Process uploaded files
        if st.button("Extract NOI Data"):
            try:
                results = handle_extraction_with_error_handling(uploaded_files)
                
                if results:
                    # Display results
                    st.success("Extraction complete!")
                    display_extraction_results(results)
                else:
                    st.warning("No results were returned. Please try again.")
                    
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}", exc_info=True)
                st.error("Oops! We hit a problem during extraction. Please try again or contact support.")
                st.error(f"Error details: {str(e)}")
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error("Something went wrong with the application. Please refresh the page and try again.")
        st.error(f"Error details: {str(e)}")
        # In a production app, you might want to hide the raw error from users 