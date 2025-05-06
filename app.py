import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import logging
import os
import json
from typing import Dict, Any, List, Optional, Tuple
import base64
from datetime import datetime
import io
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import tempfile

from utils.helpers import format_for_noi_comparison
from noi_calculations import calculate_noi_comparisons
from noi_tool_batch_integration import process_all_documents
from ai_extraction import extract_noi_data
from ai_insights_gpt import generate_insights_with_gpt
from config import get_openai_api_key, get_extraction_api_url, get_api_key, save_api_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("noi_analyzer_enhanced.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('noi_analyzer')

# Debug helper function to diagnose comparison structure issues
def debug_comparison_structure(comparison_results: Dict[str, Any]) -> None:
    """
    Debug function to analyze and log the structure of comparison results
    """
    if not comparison_results:
        logger.error("No comparison results available for debugging")
        return
        
    logger.info("=== COMPARISON RESULTS STRUCTURE DEBUG ===")
    logger.info(f"Top level keys: {list(comparison_results.keys())}")
    
    # Check for current data
    if "current" in comparison_results:
        logger.info(f"Current data keys: {list(comparison_results['current'].keys())}")
        
        # Check for financial metrics in current data
        financial_keys = ["gpr", "vacancy_loss", "other_income", "egi", "opex", "noi"]
        for key in financial_keys:
            if key in comparison_results["current"]:
                logger.info(f"Current.{key} = {comparison_results['current'][key]}")
            else:
                logger.warning(f"Missing key in current data: {key}")
    else:
        logger.warning("No 'current' key in comparison results")
    
    # Check comparison data sections
    for section in ["month_vs_prior", "actual_vs_budget", "year_vs_year"]:
        if section in comparison_results:
            logger.info(f"{section} keys: {list(comparison_results[section].keys())}")
            
            # Check for common patterns in the data
            patterns = [
                "_current",    # current values
                "_compare",    # comparison values 
                "_change",     # absolute change
                "_percent_change"  # percentage change
            ]
            
            found_patterns = {}
            for pattern in patterns:
                matches = [key for key in comparison_results[section].keys() if key.endswith(pattern)]
                found_patterns[pattern] = matches
                
                if matches:
                    logger.info(f"  Found {len(matches)} keys with pattern '{pattern}': {matches[:3]}...")
                    # Show sample values
                    if matches:
                        logger.info(f"  Sample value ({matches[0]}): {comparison_results[section][matches[0]]}")
                else:
                    logger.warning(f"  No keys found with pattern '{pattern}'")
        else:
            logger.warning(f"Missing comparison section: {section}")
    
    logger.info("=== END COMPARISON RESULTS STRUCTURE DEBUG ===")

# Set page configuration
st.set_page_config(
    page_title="NOI Analyzer Enhanced",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'current_month_actuals' not in st.session_state:
    st.session_state.current_month_actuals = None
if 'prior_month_actuals' not in st.session_state:
    st.session_state.prior_month_actuals = None
if 'current_month_budget' not in st.session_state:
    st.session_state.current_month_budget = None
if 'prior_year_actuals' not in st.session_state:
    st.session_state.prior_year_actuals = None
if 'consolidated_data' not in st.session_state:
    st.session_state.consolidated_data = None
if 'processing_completed' not in st.session_state:
    st.session_state.processing_completed = False
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'insights' not in st.session_state:
    st.session_state.insights = None
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = None
if 'show_zero_values' not in st.session_state:
    st.session_state.show_zero_values = False

# Load custom CSS
def load_css():
    with open(os.path.join(os.path.dirname(__file__), "static/css/reborn_theme.css")) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Display comparison tab
def display_comparison_tab(tab_data: Dict[str, Any], prior_key_suffix: str, name_suffix: str):
    """
    Display a comparison tab with KPI cards and detailed metrics.
    
    Args:
        tab_data: Comparison data for the tab
        prior_key_suffix: Suffix for prior period keys (e.g., 'prior', 'budget', 'prior_year')
        name_suffix: Display name for the prior period (e.g., 'Prior Month', 'Budget', 'Prior Year')
    """
    # Log detailed debugging information
    logger.info(f"Starting display_comparison_tab for {name_suffix} comparison")
    logger.info(f"tab_data keys: {list(tab_data.keys())}")
    
    # Get current data from the tab_data or from the 'current' key in comparison_results
    current_data = {}
    if "current" in tab_data:
        current_data = tab_data["current"]
        logger.info(f"Found 'current' data with keys: {list(current_data.keys())}")
    
    # Check if we have the expected data format
    has_current_format = False
    for key in ["gpr", "vacancy_loss", "other_income", "egi", "opex", "noi"]:
        if key in current_data:
            has_current_format = True
            break
    
    logger.info(f"Current data has standard format: {has_current_format}")
    
    # If we don't have the expected format, we need to extract data from the comparison structure
    if not has_current_format and any(f"{key}_current" in tab_data for key in ["gpr", "vacancy_loss", "other_income", "egi", "opex", "noi"]):
        logger.info(f"Using alternative data format for {name_suffix} comparison")
        for key in ["gpr", "vacancy_loss", "other_income", "egi", "opex", "noi"]:
            if f"{key}_current" in tab_data:
                current_data[key] = tab_data.get(f"{key}_current", 0.0)
        
        logger.info(f"Extracted current data with keys: {list(current_data.keys())}")
    
    if not current_data:
        logger.warning(f"No current data available for {name_suffix} comparison")
        st.warning(f"No current data available for {name_suffix} comparison.")
        return
        
    # Log the data structure for debugging
    logger.info(f"display_comparison_tab processing {name_suffix} comparison")
    logger.info(f"current_data keys: {list(current_data.keys())}")
    logger.info(f"current_data NOI value: {current_data.get('noi', 'Not found')}")
    
    # Look for prior values in different formats
    prior_keys = [f"noi_{prior_key_suffix}", "noi_compare", "noi_budget", "noi_prior", "noi_prior_year"]
    found_prior_keys = [key for key in prior_keys if key in tab_data]
    logger.info(f"Looking for prior NOI in keys: {prior_keys}")
    logger.info(f"Found prior NOI keys: {found_prior_keys}")
    
    if found_prior_keys:
        logger.info(f"Prior NOI value ({found_prior_keys[0]}): {tab_data.get(found_prior_keys[0], 'Not found')}")
    
    # Create columns for KPI cards
    col1, col2, col3 = st.columns(3)
    
    # Display KPI cards using Streamlit's metric component instead of custom HTML
    with col1:
        # Current value
        current_noi = current_data.get("noi", 0.0)
        st.metric(label="Current", value=f"${current_noi:,.0f}")
        
    with col2:
        # Prior period value - handle both formats (_prior_key_suffix or _compare)
        prior_noi = tab_data.get(f"noi_{prior_key_suffix}", tab_data.get("noi_compare", 0.0))
        st.metric(label=f"{name_suffix}", value=f"${prior_noi:,.0f}")
        
    with col3:
        # Change - handle both formats
        change_val = tab_data.get("noi_change", tab_data.get("noi_variance", 0.0))
        percent_change = tab_data.get("noi_percent_change", tab_data.get("noi_percent_variance", 0.0))
        
        # Format the percentage value with sign
        percent_display = f"{percent_change:.1f}%"
        if percent_change > 0:
            percent_display = f"+{percent_display}"
        
        # NOI is a positive business impact when it increases (green for increase, red for decrease)
        st.metric(
            label="Change",
            value=percent_display,
            delta=f"${change_val:,.0f}",
            delta_color="normal" if change_val >= 0 else "inverse"
        )

    # Log additional information before creating DataFrame
    logger.info(f"Creating DataFrame for {name_suffix} comparison")
    
    # Create DataFrame for data
    metrics = ["GPR", "Vacancy Loss", "Other Income", "EGI", "Total OpEx", "NOI"]
    data_keys = ["gpr", "vacancy_loss", "other_income", "egi", "opex", "noi"]

    df_data = []
    for key, name in zip(data_keys, metrics):
        # Handle both formats for current, prior, and change values
        current_val = current_data.get(key, tab_data.get(f"{key}_current", 0.0))
        prior_val = tab_data.get(f"{key}_{prior_key_suffix}", tab_data.get(f"{key}_compare", 0.0))
        change_val = tab_data.get(f"{key}_change", tab_data.get(f"{key}_variance", 0.0))
        percent_change = tab_data.get(f"{key}_percent_change", tab_data.get(f"{key}_percent_variance", 0.0))
        
        # Debug logging for data extraction
        logger.debug(f"Metric: {name}, Current: {current_val}, Prior: {prior_val}, Change: {change_val}, %Change: {percent_change}")
        
        # Skip zero values if show_zero_values is False
        if not st.session_state.show_zero_values and current_val == 0 and prior_val == 0:
            continue
            
        df_data.append({
            "Metric": name,
            "Current": current_val,
            name_suffix: prior_val,
            "Change ($)": change_val,
            "Change (%)": percent_change,
            # Add business impact direction for proper color coding
            "Direction": "inverse" if name in ["Vacancy Loss", "Total OpEx"] else "normal"
        })

    # Create DataFrame for display
    if not df_data:
        logger.warning(f"No data available for {name_suffix} display")
        st.info("No data available for display. Try enabling 'Show Zero Values' in the sidebar.")
        return
        
    logger.info(f"Created df_data with {len(df_data)} rows for {name_suffix} comparison")
    
    try:
        # Create the DataFrame
        df = pd.DataFrame(df_data)
        logger.info(f"Created DataFrame with columns: {list(df.columns)}")
        
        # Check if the DataFrame has the expected columns
        required_columns = ["Metric", "Current", name_suffix, "Change ($)", "Change (%)"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"DataFrame is missing columns: {missing_columns}")
            st.error(f"Error: DataFrame is missing required columns: {', '.join(missing_columns)}")
            # Display the raw DataFrame as a fallback
            st.dataframe(df, use_container_width=True)
            return

        # Format DataFrame for display
        df_display = df.copy()
        df_display["Current"] = df_display["Current"].apply(lambda x: f"${x:,.2f}")
        df_display[name_suffix] = df_display[name_suffix].apply(lambda x: f"${x:,.2f}")
        df_display["Change ($)"] = df_display["Change ($)"].apply(lambda x: f"${x:,.2f}")
        df_display["Change (%)"] = df_display["Change (%)"].apply(lambda x: f"{x:.1f}%")
        
        # Create styleable dataframe with business-appropriate color coding
        df_styled = df_display.drop(columns=["Direction"])
        
        # Apply conditional formatting based on business impact
        def style_df(row):
            styles = [''] * len(row)
            
            # Get indices of the columns to style
            pct_change_idx = list(row.index).index("Change (%)")
            dollar_change_idx = list(row.index).index("Change ($)")
            
            direction = row["Direction"]
            change_pct = float(row["Change (%)"].strip('%'))
            
            # Determine if the change is positive from a business perspective
            if direction == "inverse":
                # For metrics where decrease is good (Vacancy Loss, OpEx)
                is_positive = change_pct < 0
            else:
                # For metrics where increase is good (NOI, GPR, etc.)
                is_positive = change_pct > 0
                
            # Apply appropriate colors
            color = "color: green" if is_positive else "color: red" if change_pct != 0 else ""
            
            # Apply to both dollar and percentage columns
            styles[pct_change_idx] = color
            styles[dollar_change_idx] = color
            
            return styles
        
        # Apply styling and display
        styled_df = df_display.style.apply(style_df, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Add OpEx Breakdown expander section
        with st.expander("Operating Expense Breakdown"):
            # Check if we have OpEx component data
            opex_components = ["property_taxes", "insurance", "repairs_and_maintenance", "utilities", "management_fees"]
            
            if any(component in current_data for component in opex_components):
                # Create DataFrame for OpEx components
                opex_df_data = []
                opex_metrics = ["Property Taxes", "Insurance", "Repairs & Maintenance", "Utilities", "Management Fees"]
                
                for key, name in zip(opex_components, opex_metrics):
                    # Handle both formats for current, prior, and change values
                    current_val = current_data.get(key, tab_data.get(f"{key}_current", 0.0))
                    prior_val = tab_data.get(f"{key}_{prior_key_suffix}", tab_data.get(f"{key}_compare", 0.0))
                    change_val = tab_data.get(f"{key}_change", tab_data.get(f"{key}_variance", 0.0))
                    percent_change = tab_data.get(f"{key}_percent_change", tab_data.get(f"{key}_percent_variance", 0.0))
                    
                    # Skip zero values if show_zero_values is False
                    if not st.session_state.show_zero_values and current_val == 0 and prior_val == 0:
                        continue
                        
                    opex_df_data.append({
                        "Expense Category": name,
                        "Current": current_val,
                        name_suffix: prior_val,
                        "Change ($)": change_val,
                        "Change (%)": percent_change,
                        # Operating expenses are positive when they decrease
                        "Direction": "inverse"
                    })
                
                if opex_df_data:
                    opex_df = pd.DataFrame(opex_df_data)
                    
                    # Format DataFrame for display
                    opex_df_display = opex_df.copy()
                    opex_df_display["Current"] = opex_df_display["Current"].apply(lambda x: f"${x:,.2f}")
                    opex_df_display[name_suffix] = opex_df_display[name_suffix].apply(lambda x: f"${x:,.2f}")
                    opex_df_display["Change ($)"] = opex_df_display["Change ($)"].apply(lambda x: f"${x:,.2f}")
                    opex_df_display["Change (%)"] = opex_df_display["Change (%)"].apply(lambda x: f"{x:.1f}%")
                    
                    # Apply styling and display
                    opex_styled_df = opex_df_display.drop(columns=["Direction"])
                    
                    # Create a function to apply styling
                    def style_opex_df(row):
                        styles = [''] * len(row)
                        
                        # Get indices of the columns to style
                        pct_change_idx = list(row.index).index("Change (%)")
                        dollar_change_idx = list(row.index).index("Change ($)")
                        
                        change_pct = float(row["Change (%)"].strip('%'))
                        
                        # For expenses, a decrease (negative change) is good
                        is_positive = change_pct < 0
                        
                        # Apply appropriate colors
                        color = "color: green" if is_positive else "color: red" if change_pct != 0 else ""
                        
                        # Apply to both dollar and percentage columns
                        styles[pct_change_idx] = color
                        styles[dollar_change_idx] = color
                        
                        return styles
                    
                    # Apply styling and display
                    opex_styled = opex_df_display.style.apply(style_opex_df, axis=1)
                    st.dataframe(opex_styled, use_container_width=True)
                    
                    # Display a pie chart to visualize the breakdown
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Filter out zero values for the pie chart
                        pie_data = opex_df[opex_df["Current"] > 0]
                        if not pie_data.empty:
                            fig = px.pie(
                                pie_data, 
                                values="Current", 
                                names="Expense Category",
                                title="Current Operating Expenses Breakdown",
                                color_discrete_sequence=px.colors.qualitative.Set3,
                                hole=0.4
                            )
                            fig.update_layout(
                                template="plotly_dark",
                                plot_bgcolor='rgba(30, 41, 59, 0.8)',
                                paper_bgcolor='rgba(16, 23, 42, 0)',
                                font=dict(
                                    family="Inter, sans-serif",
                                    size=12,
                                    color="#F0F0F0"
                                ),
                                title_font=dict(size=16, color="#F0F0F0", family="Inter, sans-serif"),
                                legend=dict(font=dict(size=10, color="#F0F0F0"))
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Create a horizontal bar chart for comparison
                        if not opex_df.empty:
                            comp_fig = go.Figure()
                            
                            # Add current period bars
                            comp_fig.add_trace(go.Bar(
                                y=opex_df["Expense Category"],
                                x=opex_df["Current"],
                                name="Current",
                                orientation='h',
                                marker=dict(
                                    color='rgba(13, 110, 253, 0.8)',
                                    line=dict(width=1, color='white')
                                )
                            ))
                            
                            # Add prior period bars
                            comp_fig.add_trace(go.Bar(
                                y=opex_df["Expense Category"],
                                x=opex_df[name_suffix],
                                name=name_suffix,
                                orientation='h',
                                marker=dict(
                                    color='rgba(32, 201, 151, 0.8)',
                                    line=dict(width=1, color='white')
                                )
                            ))
                            
                            # Update layout
                            comp_fig.update_layout(
                                title=f"OpEx Components: Current vs {name_suffix}",
                                barmode='group',
                                xaxis_title="Amount ($)",
                                xaxis=dict(tickprefix="$"),
                                template="plotly_dark",
                                plot_bgcolor='rgba(30, 41, 59, 0.8)',
                                paper_bgcolor='rgba(16, 23, 42, 0)',
                                font=dict(
                                    family="Inter, sans-serif",
                                    size=12,
                                    color="#F0F0F0"
                                ),
                                title_font=dict(size=16, color="#F0F0F0", family="Inter, sans-serif"),
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=-0.2,
                                    xanchor="center",
                                    x=0.5,
                                    font=dict(size=10, color="#F0F0F0")
                                )
                            )
                            
                            st.plotly_chart(comp_fig, use_container_width=True)
                else:
                    st.info("No operating expense details available for this period.")
            else:
                st.info("Operating expense breakdown is not available for this comparison.")
                
        # Add Other Income Breakdown expander section
        with st.expander("Other Income Breakdown"):
            # Check if we have Other Income component data
            other_income_components = ["parking", "laundry", "late_fees", "pet_fees", "application_fees"]
            
            if any(component in current_data for component in other_income_components):
                # Create DataFrame for Other Income components
                income_df_data = []
                income_metrics = ["Parking", "Laundry", "Late Fees", "Pet Fees", "Application Fees"]
                
                for key, name in zip(other_income_components, income_metrics):
                    # Handle both formats for current, prior, and change values
                    current_val = current_data.get(key, tab_data.get(f"{key}_current", 0.0))
                    prior_val = tab_data.get(f"{key}_{prior_key_suffix}", tab_data.get(f"{key}_compare", 0.0))
                    change_val = tab_data.get(f"{key}_change", tab_data.get(f"{key}_variance", 0.0))
                    percent_change = tab_data.get(f"{key}_percent_change", tab_data.get(f"{key}_percent_variance", 0.0))
                    
                    # Skip zero values if show_zero_values is False
                    if not st.session_state.show_zero_values and current_val == 0 and prior_val == 0:
                        continue
                        
                    income_df_data.append({
                        "Income Category": name,
                        "Current": current_val,
                        name_suffix: prior_val,
                        "Change ($)": change_val,
                        "Change (%)": percent_change,
                        # Other Income is positive when it increases
                        "Direction": "normal"
                    })
                
                if income_df_data:
                    income_df = pd.DataFrame(income_df_data)
                    
                    # Format DataFrame for display
                    income_df_display = income_df.copy()
                    income_df_display["Current"] = income_df_display["Current"].apply(lambda x: f"${x:,.2f}")
                    income_df_display[name_suffix] = income_df_display[name_suffix].apply(lambda x: f"${x:,.2f}")
                    income_df_display["Change ($)"] = income_df_display["Change ($)"].apply(lambda x: f"${x:,.2f}")
                    income_df_display["Change (%)"] = income_df_display["Change (%)"].apply(lambda x: f"{x:.1f}%")
                    
                    # Apply styling and display
                    income_styled_df = income_df_display.drop(columns=["Direction"])
                    
                    # Create a function to apply styling
                    def style_income_df(row):
                        styles = [''] * len(row)
                        
                        # Get indices of the columns to style
                        pct_change_idx = list(row.index).index("Change (%)")
                        dollar_change_idx = list(row.index).index("Change ($)")
                        
                        change_pct = float(row["Change (%)"].strip('%'))
                        
                        # For income, an increase (positive change) is good
                        is_positive = change_pct > 0
                        
                        # Apply appropriate colors
                        color = "color: green" if is_positive else "color: red" if change_pct != 0 else ""
                        
                        # Apply to both dollar and percentage columns
                        styles[pct_change_idx] = color
                        styles[dollar_change_idx] = color
                        
                        return styles
                    
                    # Apply styling and display
                    income_styled = income_df_display.style.apply(style_income_df, axis=1)
                    st.dataframe(income_styled, use_container_width=True)
                    
                    # Display a pie chart to visualize the breakdown
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Filter out zero values for the pie chart
                        pie_data = income_df[income_df["Current"] > 0]
                        if not pie_data.empty:
                            fig = px.pie(
                                pie_data, 
                                values="Current", 
                                names="Income Category",
                                title="Current Other Income Breakdown",
                                color_discrete_sequence=px.colors.qualitative.Pastel,
                                hole=0.4
                            )
                            fig.update_layout(
                                template="plotly_dark",
                                plot_bgcolor='rgba(30, 41, 59, 0.8)',
                                paper_bgcolor='rgba(16, 23, 42, 0)',
                                font=dict(
                                    family="Inter, sans-serif",
                                    size=12,
                                    color="#F0F0F0"
                                ),
                                title_font=dict(size=16, color="#F0F0F0", family="Inter, sans-serif"),
                                legend=dict(font=dict(size=10, color="#F0F0F0"))
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Create comparison bar chart
                        comp_data = income_df.copy()
                        
                        # Sort by current value
                        comp_data = comp_data.sort_values(by="Current", ascending=True)
                        
                        # Create horizontal bar chart for comparing current vs prior
                        comp_fig = go.Figure()
                        
                        # Add bars for current and prior values
                        comp_fig.add_trace(go.Bar(
                            y=comp_data["Income Category"],
                            x=comp_data["Current"],
                            name="Current",
                            orientation='h',
                            marker=dict(color="#4DB6AC")
                        ))
                        
                        comp_fig.add_trace(go.Bar(
                            y=comp_data["Income Category"],
                            x=comp_data[name_suffix],
                            name=name_suffix,
                            orientation='h',
                            marker=dict(color="#BA68C8")
                        ))
                        
                        # Update layout
                        comp_fig.update_layout(
                            title=f"Other Income Components: Current vs {name_suffix}",
                            barmode='group',
                            xaxis_title="Amount ($)",
                            xaxis=dict(tickprefix="$"),
                            template="plotly_dark",
                            plot_bgcolor='rgba(30, 41, 59, 0.8)',
                            paper_bgcolor='rgba(16, 23, 42, 0)',
                            font=dict(
                                family="Inter, sans-serif",
                                size=12,
                                color="#F0F0F0"
                            ),
                            title_font=dict(size=16, color="#F0F0F0", family="Inter, sans-serif"),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=-0.2,
                                xanchor="center",
                                x=0.5,
                                font=dict(size=10, color="#F0F0F0")
                            )
                        )
                        
                        st.plotly_chart(comp_fig, use_container_width=True)
                else:
                    st.info("No other income details available for this period.")
            else:
                st.info("Other income breakdown is not available for this comparison.")
    except KeyError as e:
        logger.error(f"KeyError in DataFrame operation: {str(e)}")
        st.error(f"Error accessing DataFrame column: {str(e)}")
        # Display the raw data as a fallback
        st.json(df_data)
        return
    except Exception as e:
        logger.error(f"Error formatting DataFrame: {str(e)}")
        st.error(f"Error formatting data for display: {str(e)}")
        # Display the raw data as a fallback
        st.json(df_data)
        return

    try:
        logger.info(f"Creating charts for {name_suffix} comparison")
        # Create bar chart for visual comparison
        fig = go.Figure()

        # Calculate change percentages for hover data
        change_pcts = df["Change (%)"].tolist()
        change_vals = df["Change ($)"].tolist()
        directions = df["Direction"].tolist()
        metrics = df["Metric"].tolist()
        
        # Helper function to determine if a change is positive from business perspective
        def is_positive_change(metric, change_val):
            if metric in ["Vacancy Loss", "Total OpEx"]:
                # For these metrics, a decrease (negative change) is good
                return change_val < 0
            else:
                # For other metrics (NOI, GPR, EGI, etc.), an increase (positive change) is good
                return change_val > 0
        
        # Create custom color scales based on values to add visual contrast
        current_colors = []
        compare_colors = []
        
        # Use a gradient of blue shades for current values
        current_max = max(df["Current"]) if not df["Current"].empty else 0
        for val in df["Current"]:
            intensity = min(1.0, 0.3 + 0.7 * (val / current_max)) if current_max > 0 else 0.5
            current_colors.append(f'rgba(13, 110, 253, {intensity})')
            
        # Use a gradient of teal shades for comparison values
        compare_max = max(df[name_suffix]) if not df[name_suffix].empty else 0
        for val in df[name_suffix]:
            intensity = min(1.0, 0.3 + 0.7 * (val / compare_max)) if compare_max > 0 else 0.5
            compare_colors.append(f'rgba(32, 201, 151, {intensity})')

        # Add current period bars with enhanced styling
        fig.add_trace(go.Bar(
            x=df["Metric"],
            y=df["Current"],
            name="Current",
            marker=dict(
                color=current_colors,
                line=dict(width=1, color='white')
            ),
            opacity=0.9,
            customdata=list(zip(change_vals, change_pcts, directions)),
            hovertemplate='<b>%{x}</b><br>Current: $%{y:,.0f}<br>Change: $%{customdata[0]:,.0f} (%{customdata[1]:.1f}%)<extra></extra>'
        ))

        # Add prior period bars with enhanced styling
        fig.add_trace(go.Bar(
            x=df["Metric"],
            y=df[name_suffix],
            name=name_suffix,
            marker=dict(
                color=compare_colors,
                line=dict(width=1, color='white')
            ),
            opacity=0.9,
            customdata=list(zip(df["Metric"])),
            hovertemplate='<b>%{x}</b><br>' + f'{name_suffix}: $' + '%{y:,.0f}<extra></extra>'
        ))

        # Identify peak NOI for annotation
        if 'NOI' in df['Metric'].values:
            current_noi = df.loc[df['Metric'] == 'NOI', 'Current'].values[0]
            prior_noi = df.loc[df['Metric'] == 'NOI', name_suffix].values[0]
            
            # Calculate the difference and percentage change
            noi_diff = current_noi - prior_noi
            noi_pct = (noi_diff / prior_noi * 100) if prior_noi != 0 else 0
            
            # Create annotation text based on whether NOI increased or decreased
            # For NOI, an increase is positive (green), decrease is negative (red)
            if noi_diff > 0:
                annotation_text = f"NOI increased by ${noi_diff:,.0f}<br>({noi_pct:.1f}%)"
                arrow_color = "green"
            elif noi_diff < 0:
                annotation_text = f"NOI decreased by ${abs(noi_diff):,.0f}<br>({noi_pct:.1f}%)"
                arrow_color = "red"
            else:
                annotation_text = "NOI unchanged"
                arrow_color = "gray"
                
            # Add annotation for NOI
            fig.add_annotation(
                x='NOI', 
                y=max(current_noi, prior_noi) * 1.1,
                text=annotation_text,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=arrow_color,
                bgcolor="rgba(30, 41, 59, 0.8)",
                bordercolor=arrow_color,
                borderwidth=1,
                borderpad=4,
                font=dict(color="#F0F0F0", size=12)
            )

        # Update layout with dark theme styling
        fig.update_layout(
            barmode='group',
            title=f"Current vs {name_suffix}",
            title_font=dict(size=18, color="#F0F0F0", family="Inter, sans-serif"),
            template="plotly_dark",
            plot_bgcolor='rgba(30, 41, 59, 0.8)',
            paper_bgcolor='rgba(16, 23, 42, 0)',
            font=dict(
                family="Inter, sans-serif",
                size=14,
                color="#F0F0F0"
            ),
            margin=dict(l=20, r=20, t=60, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(10, 15, 30, 0.6)",
                bordercolor="rgba(255, 255, 255, 0.1)",
                borderwidth=1,
                font=dict(
                    color="#F0F0F0"
                )
            ),
            xaxis=dict(
                title="",
                tickfont=dict(size=14),
                showgrid=False,
                zeroline=False,
                color="#F0F0F0"
            ),
            yaxis=dict(
                title="Amount ($)",
                titlefont=dict(size=14, color="#F0F0F0"),
                tickfont=dict(size=12, color="#F0F0F0"),
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                zeroline=False
            ),
            hoverlabel=dict(
                bgcolor="#1E293B",
                font_size=14,
                font_family="Inter, sans-serif",
                font_color="#F0F0F0"
            )
        )

        # Add dollar sign to y-axis labels
        fig.update_yaxes(tickprefix="$")
        
        # Add subtle pattern and depth to bars
        for trace in fig.data:
            trace.update(
                marker_pattern_shape="",
                marker_line_width=1,
                marker_line_color="white"
            )

        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        logger.info(f"Successfully displayed chart for {name_suffix} comparison")
    except Exception as e:
        logger.error(f"Error creating chart: {str(e)}")
        st.error(f"Error creating chart: {str(e)}")


# Display NOI comparisons
def display_noi_comparisons(comparison_results: Dict[str, Any]):
    """
    Display NOI comparisons with tabs for different comparison types.
    
    Args:
        comparison_results: Results from calculate_noi_comparisons()
    """
    if not comparison_results:
        st.warning("No comparison results available.")
        return
        
    # Log the comparison results structure for debugging
    logger.info(f"display_noi_comparisons called with keys: {list(comparison_results.keys())}")
    
    st.markdown('<h2 class="section-header">Comparative Analysis</h2>', unsafe_allow_html=True)
    
    # Create tabs for different comparison types
    tab1, tab2, tab3 = st.tabs(["Current vs Budget", "Current vs Prior Year", "Current vs Prior Month"])
    
    # Display comparison tabs
    with tab1:
        if "actual_vs_budget" in comparison_results:
            try:
                # Add current data to the actual_vs_budget dict if it doesn't have it
                budget_data = comparison_results["actual_vs_budget"].copy()
                if "current" not in budget_data and "current" in comparison_results:
                    budget_data["current"] = comparison_results["current"]
                    
                display_comparison_tab(budget_data, "budget", "Budget")
            except Exception as e:
                logger.error(f"Error in Budget comparison tab: {str(e)}")
                st.error(f"Error displaying Budget comparison: {str(e)}")
        else:
            st.info("No budget data available for comparison.")
    with tab2:
        if "year_vs_year" in comparison_results:
            try:
                # Add current data to the year_vs_year dict if it doesn't have it
                yoy_data = comparison_results["year_vs_year"].copy()
                if "current" not in yoy_data and "current" in comparison_results:
                    yoy_data["current"] = comparison_results["current"]
                    
                display_comparison_tab(yoy_data, "prior_year", "Prior Year")
            except Exception as e:
                logger.error(f"Error in Prior Year comparison tab: {str(e)}")
                st.error(f"Error displaying Prior Year comparison: {str(e)}")
        else:
            st.info("No prior year data available for comparison.")
    with tab3:
        if "month_vs_prior" in comparison_results:
            try:
                # Add current data to the month_vs_prior dict if it doesn't have it
                mom_data = comparison_results["month_vs_prior"].copy()
                if "current" not in mom_data and "current" in comparison_results:
                    mom_data["current"] = comparison_results["current"]
                    
                display_comparison_tab(mom_data, "prior", "Prior Month")
            except Exception as e:
                logger.error(f"Error in Prior Month comparison tab: {str(e)}")
                st.error(f"Error displaying Prior Month comparison: {str(e)}")
        else:
            st.info("No prior month data available for comparison.")


def generate_pdf_report(comparison_results: Dict[str, Any], property_name: str = "Property") -> Optional[str]:
    """
    Generate a PDF report from the comparison results.
    
    Args:
        comparison_results: Results from calculate_noi_comparisons()
        property_name: Name of the property
        
    Returns:
        Path to the generated PDF file or None if generation failed
    """
    try:
        logger.info("Generating PDF report")
        
        # Extract KPIs from comparison results
        current_data = comparison_results.get("current", {})
        
        if not current_data:
            logger.warning("No current data available for PDF report")
            return None
            
        # Calculate some additional KPIs
        egi = current_data.get("egi", 0)
        opex = current_data.get("opex", 0)
        noi = current_data.get("noi", 0)
        
        operating_expense_ratio = opex / egi if egi else 0
        noi_margin = noi / egi if egi else 0
        
        # Default value for gross rent multiplier
        gross_rent_multiplier = 10.0  # Example value, would be calculated from property value
        
        kpis = {
            "egi": egi,
            "opex": opex,
            "noi": noi,
            "operating_expense_ratio": operating_expense_ratio,
            "noi_margin": noi_margin,
            "gross_rent_multiplier": gross_rent_multiplier
        }
        
        # Prepare performance data for the table
        performance_data = []
        metrics = ["GPR", "Vacancy Loss", "Other Income", "EGI", "Total OpEx", "NOI"]
        data_keys = ["gpr", "vacancy_loss", "other_income", "egi", "opex", "noi"]
        
        # Find the comparison with most data - prioritize budget comparison
        comparison_section = None
        for section in ["actual_vs_budget", "year_vs_year", "month_vs_prior"]:
            if section in comparison_results:
                comparison_section = section
                break
                
        if not comparison_section:
            logger.warning("No comparison data available for PDF report")
            return None
            
        comparison_data = comparison_results[comparison_section]
        
        for key, metric in zip(data_keys, metrics):
            current_val = current_data.get(key, 0)
            
            # Look for prior value in different formats
            prior_val = 0
            for prior_key in [f"{key}_compare", f"{key}_prior", f"{key}_budget", f"{key}_prior_year"]:
                if prior_key in comparison_data:
                    prior_val = comparison_data[prior_key]
                    break
                    
            # Calculate variance
            variance = (current_val - prior_val) / prior_val if prior_val else 0
            
            performance_data.append({
                "metric": metric,
                "current": current_val,
                "prior": prior_val,
                "variance": variance
            })
        
        # Check if OpEx breakdown is available and prepare data
        opex_breakdown_available = False
        opex_breakdown_data = []
        opex_components = ["property_taxes", "insurance", "repairs_and_maintenance", "utilities", "management_fees"]
        opex_names = ["Property Taxes", "Insurance", "Repairs & Maintenance", "Utilities", "Management Fees"]
        
        if any(component in current_data for component in opex_components):
            opex_breakdown_available = True
            
            # Calculate percentages of total OpEx
            total_opex = current_data.get("opex", 0)
            
            for key, name in zip(opex_components, opex_names):
                current_val = current_data.get(key, 0)
                
                # Look for prior value in different formats
                prior_val = 0
                for prior_key in [f"{key}_compare", f"{key}_prior", f"{key}_budget", f"{key}_prior_year"]:
                    if prior_key in comparison_data:
                        prior_val = comparison_data[prior_key]
                        break
                
                # Calculate variance and percentage of total
                variance = (current_val - prior_val) / prior_val if prior_val else 0
                percentage = (current_val / total_opex) * 100 if total_opex else 0
                
                opex_breakdown_data.append({
                    "category": name,
                    "current": current_val,
                    "prior": prior_val,
                    "variance": variance,
                    "percentage": percentage
                })
        
        # Check if Other Income breakdown is available and prepare data
        income_breakdown_available = False
        income_breakdown_data = []
        other_income_components = ["parking", "laundry", "late_fees", "pet_fees", "application_fees"]
        income_names = ["Parking", "Laundry", "Late Fees", "Pet Fees", "Application Fees"]
        
        if any(component in current_data for component in other_income_components):
            income_breakdown_available = True
            
            # Calculate percentages of total Other Income
            total_income = current_data.get("other_income", 0)
            
            for key, name in zip(other_income_components, income_names):
                current_val = current_data.get(key, 0)
                
                # Look for prior value in different formats
                prior_val = 0
                for prior_key in [f"{key}_compare", f"{key}_prior", f"{key}_budget", f"{key}_prior_year"]:
                    if prior_key in comparison_data:
                        prior_val = comparison_data[prior_key]
                        break
                
                # Calculate variance and percentage of total
                variance = (current_val - prior_val) / prior_val if prior_val else 0
                percentage = (current_val / total_income) * 100 if total_income else 0
                
                income_breakdown_data.append({
                    "category": name,
                    "current": current_val,
                    "prior": prior_val,
                    "variance": variance,
                    "percentage": percentage
                })
        
        # Configure Jinja environment
        env = Environment(loader=FileSystemLoader('templates'))
        template = env.get_template('report.html')
        
        # Get insights data if available
        executive_summary = ""
        performance_insights = []
        recommendations = []
        
        if "insights" in st.session_state and st.session_state.insights:
            insights = st.session_state.insights
            executive_summary = insights.get("summary", "")
            performance_insights = insights.get("performance", [])
            recommendations = insights.get("recommendations", [])
        
        # Prepare template data
        template_data = {
            'datetime': datetime,
            'property_name': property_name,
            'kpis': kpis,
            'performance_data': performance_data,
            'executive_summary': executive_summary,
            'performance_insights': performance_insights,
            'recommendations': recommendations,
            'opex_breakdown_available': opex_breakdown_available,
            'opex_breakdown_data': opex_breakdown_data,
            'income_breakdown_available': income_breakdown_available,
            'income_breakdown_data': income_breakdown_data
        }
        
        # Render template to HTML
        html_content = template.render(**template_data)
        
        # Create a temporary file for the PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            # Generate PDF from HTML
            html_doc = HTML(string=html_content)
            pdf_bytes = html_doc.write_pdf()
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
            
        logger.info(f"PDF report generated successfully: {tmp_path}")
        return tmp_path
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        return None

def export_to_excel(comparison_results: Dict[str, Any], property_name: str = "Property") -> Optional[bytes]:
    """
    Export comparison results to Excel file.
    
    Args:
        comparison_results: Results from calculate_noi_comparisons()
        property_name: Name of the property
        
    Returns:
        Excel file as bytes or None if export failed
    """
    try:
        logger.info("Exporting data to Excel")
        
        # Create an Excel writer
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Current Data Sheet
            if "current" in comparison_results:
                current_data = comparison_results["current"]
                df_current = pd.DataFrame({
                    "Metric": ["GPR", "Vacancy Loss", "Other Income", "EGI", "Total OpEx", "NOI"],
                    "Value": [
                        current_data.get("gpr", 0),
                        current_data.get("vacancy_loss", 0),
                        current_data.get("other_income", 0),
                        current_data.get("egi", 0),
                        current_data.get("opex", 0),
                        current_data.get("noi", 0)
                    ]
                })
                df_current.to_excel(writer, sheet_name="Current Data", index=False)
                
                # Format the sheet
                workbook = writer.book
                worksheet = writer.sheets["Current Data"]
                money_format = workbook.add_format({'num_format': '$#,##0.00'})
                worksheet.set_column('B:B', 15, money_format)
                
                # Add OpEx Breakdown sheet if data is available
                opex_components = ["property_taxes", "insurance", "repairs_and_maintenance", "utilities", "management_fees"]
                if any(component in current_data for component in opex_components):
                    opex_data = {
                        "Category": ["Property Taxes", "Insurance", "Repairs & Maintenance", "Utilities", "Management Fees", "Total OpEx"],
                        "Amount": [
                            current_data.get("property_taxes", 0),
                            current_data.get("insurance", 0),
                            current_data.get("repairs_and_maintenance", 0),
                            current_data.get("utilities", 0),
                            current_data.get("management_fees", 0),
                            current_data.get("opex", 0)
                        ],
                        "Percent of Total": []
                    }
                    
                    # Calculate percentages
                    total_opex = current_data.get("opex", 0)
                    if total_opex > 0:
                        for component_value in opex_data["Amount"][:-1]:  # Exclude the total itself
                            percent = (component_value / total_opex) * 100 if total_opex else 0
                            opex_data["Percent of Total"].append(percent)
                        opex_data["Percent of Total"].append(100)  # Total is 100%
                    else:
                        opex_data["Percent of Total"] = [0, 0, 0, 0, 0, 100]
                    
                    # Create DataFrame and write to Excel
                    df_opex = pd.DataFrame(opex_data)
                    df_opex.to_excel(writer, sheet_name="OpEx Breakdown", index=False)
                    
                    # Format the OpEx breakdown sheet
                    worksheet_opex = writer.sheets["OpEx Breakdown"]
                    percent_format = workbook.add_format({'num_format': '0.00%'})
                    worksheet_opex.set_column('B:B', 15, money_format)
                    worksheet_opex.set_column('C:C', 15, percent_format)
            
            # Add Other Income breakdown if available
            other_income_components = ["parking", "laundry", "late_fees", "pet_fees", "application_fees"]
            if any(component in current_data for component in other_income_components):
                income_data = {
                    "Category": ["Parking", "Laundry", "Late Fees", "Pet Fees", "Application Fees", "Total Other Income"],
                    "Amount": [
                        current_data.get("parking", 0),
                        current_data.get("laundry", 0),
                        current_data.get("late_fees", 0),
                        current_data.get("pet_fees", 0),
                        current_data.get("application_fees", 0),
                        current_data.get("other_income", 0)
                    ],
                    "Percent of Total": []
                }
                
                # Calculate percentages
                total_income = current_data.get("other_income", 0)
                if total_income > 0:
                    for component_value in income_data["Amount"][:-1]:  # Exclude the total itself
                        percent = (component_value / total_income) * 100 if total_income else 0
                        income_data["Percent of Total"].append(percent)
                    income_data["Percent of Total"].append(100)  # Total is 100%
                else:
                    income_data["Percent of Total"] = [0, 0, 0, 0, 0, 100]
                
                # Create DataFrame and write to Excel
                df_income = pd.DataFrame(income_data)
                df_income.to_excel(writer, sheet_name="Other Income Breakdown", index=False)
                
                # Format the Other Income breakdown sheet
                worksheet_income = writer.sheets["Other Income Breakdown"]
                worksheet_income.set_column('B:B', 15, money_format)
                worksheet_income.set_column('C:C', 15, percent_format)
            
            # Create sheets for each comparison type
            for comparison_type, title in [
                ("actual_vs_budget", "Actual vs Budget"),
                ("year_vs_year", "Year vs Year"),
                ("month_vs_prior", "Month vs Prior Month")
            ]:
                if comparison_type in comparison_results:
                    comparison_data = comparison_results[comparison_type]
                    
                    # Create DataFrame for the comparison
                    df_data = []
                    metrics = ["GPR", "Vacancy Loss", "Other Income", "EGI", "Total OpEx", "NOI"]
                    data_keys = ["gpr", "vacancy_loss", "other_income", "egi", "opex", "noi"]
                    
                    for key, name in zip(data_keys, metrics):
                        # Handle both formats for current, prior, and change values
                        current_val = comparison_data.get(f"{key}_current", 0.0)
                        compare_val = comparison_data.get(f"{key}_compare", 0.0)
                        change_val = comparison_data.get(f"{key}_change", 0.0)
                        percent_change = comparison_data.get(f"{key}_percent_change", 0.0)
                        
                        df_data.append({
                            "Metric": name,
                            "Current": current_val,
                            "Comparison": compare_val,
                            "Change ($)": change_val,
                            "Change (%)": percent_change
                        })
                    
                    # Create DataFrame and write to Excel
                    df_comparison = pd.DataFrame(df_data)
                    df_comparison.to_excel(writer, sheet_name=title, index=False)
                    
                    # Format the sheet
                    worksheet = writer.sheets[title]
                    money_format = workbook.add_format({'num_format': '$#,##0.00'})
                    percent_format = workbook.add_format({'num_format': '0.00%'})
                    
                    worksheet.set_column('B:D', 15, money_format)
                    worksheet.set_column('E:E', 15, percent_format)
                    
                    # Add OpEx component comparison if data is available
                    opex_components = ["property_taxes", "insurance", "repairs_and_maintenance", "utilities", "management_fees"]
                    opex_names = ["Property Taxes", "Insurance", "Repairs & Maintenance", "Utilities", "Management Fees"]
                    
                    if any(f"{component}_current" in comparison_data for component in opex_components):
                        sheet_name = f"{title} - OpEx Detail"
                        
                        # Create DataFrame for OpEx component comparison
                        opex_comparison_data = []
                        
                        for key, name in zip(opex_components, opex_names):
                            # Handle both formats for current, prior, and change values
                            current_val = comparison_data.get(f"{key}_current", 0.0)
                            compare_val = comparison_data.get(f"{key}_compare", 0.0)
                            change_val = comparison_data.get(f"{key}_change", 0.0)
                            percent_change = comparison_data.get(f"{key}_percent_change", 0.0)
                            
                            opex_comparison_data.append({
                                "OpEx Category": name,
                                "Current": current_val,
                                "Comparison": compare_val,
                                "Change ($)": change_val,
                                "Change (%)": percent_change
                            })
                        
                        # Add total row
                        opex_comparison_data.append({
                            "OpEx Category": "Total OpEx",
                            "Current": comparison_data.get("opex_current", 0.0),
                            "Comparison": comparison_data.get("opex_compare", 0.0),
                            "Change ($)": comparison_data.get("opex_change", 0.0),
                            "Change (%)": comparison_data.get("opex_percent_change", 0.0)
                        })
                        
                        # Create DataFrame and write to Excel
                        df_opex_comparison = pd.DataFrame(opex_comparison_data)
                        df_opex_comparison.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        # Format the sheet
                        worksheet_opex_detail = writer.sheets[sheet_name]
                        worksheet_opex_detail.set_column('B:D', 15, money_format)
                        worksheet_opex_detail.set_column('E:E', 15, percent_format)
            
            # If insights are available, add them to the Excel file
            if "insights" in st.session_state and st.session_state.insights:
                insights = st.session_state.insights
                
                # Create insights sheet
                insights_data = []
                
                # Add summary
                if "summary" in insights:
                    insights_data.append({"Category": "Summary", "Content": insights["summary"]})
                
                # Add performance insights
                if "performance" in insights and insights["performance"]:
                    for i, insight in enumerate(insights["performance"], 1):
                        insights_data.append({"Category": f"Performance Insight {i}", "Content": insight})
                
                # Add recommendations
                if "recommendations" in insights and insights["recommendations"]:
                    for i, rec in enumerate(insights["recommendations"], 1):
                        insights_data.append({"Category": f"Recommendation {i}", "Content": rec})
                
                # Create DataFrame and write to Excel
                df_insights = pd.DataFrame(insights_data)
                df_insights.to_excel(writer, sheet_name="AI Insights", index=False)
                
                # Format the insights sheet
                worksheet = writer.sheets["AI Insights"]
                worksheet.set_column('A:A', 20)
                worksheet.set_column('B:B', 70)
        
        # Get the Excel file as bytes
        output.seek(0)
        excel_data = output.getvalue()
        
        logger.info("Excel export completed successfully")
        return excel_data
        
    except Exception as e:
        logger.error(f"Error exporting to Excel: {str(e)}")
        return None
