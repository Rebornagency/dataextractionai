"""
NOI Tool - Combined Module (Enhanced Frontend for Detailed Analysis)
This file combines all modules from the NOI Tool project into a single file.
This version is updated to work with the enhanced backend API that provides
detailed financial data (GPR, Vacancy, EGI, OpEx, Reserves, Application Fees).

Original modules:
- app.py: Main Streamlit application for NOI Analyzer
- noi_calculations.py: Calculates NOI comparisons based on consolidated data
- config.py: Configuration settings and API keys
- noi_tool_batch_integration.py: Handles batch processing of documents
- ai_insights_gpt.py: Generates insights using GPT
- insights_display.py: Displays insights in the Streamlit UI
- ai_extraction.py: Extracts financial data from documents
- utils/helpers.py: Utility functions for data formatting

Enhancements:
- Handles detailed financial data structure from enhanced backend API.
- Calculates comparisons based on GPR, EGI, Vacancy, OpEx.
- Displays more detailed metrics.
- Provides richer context for AI insights.

Dependencies (ensure these are installed):
- streamlit>=1.32.0
- pandas>=2.1.4
- numpy>=1.26.3
- requests>=2.31.0
- plotly>=5.18.0
- python-dotenv>=1.0.0
- openai>=1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import logging
import time
import traceback
from typing import Dict, Any, List, Union, Optional, Tuple
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import math # Import math for isnan checks

# --- Configuration Section ---
# Configure logging
logging.basicConfig(
    level=logging.INFO, # Keep frontend logs at INFO unless debugging frontend itself
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("noi_analyzer_frontend.log"), # New log file name
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('noi_analyzer_frontend')

# --- Default API Configuration (User can override in Settings) ---

# Function to get OpenAI API key (prioritize session state, then env var, then hardcoded default)
def get_openai_api_key() -> str:
    if "openai_api_key" in st.session_state and st.session_state.openai_api_key:
        return st.session_state.openai_api_key
    env_key = os.environ.get("OPENAI_API_KEY", "")
    if env_key:
        st.session_state.openai_api_key = env_key # Store in session state if found
        return env_key
    # Avoid hardcoding secrets, return empty if not found
    st.session_state.openai_api_key = ""
    return ""

# Function to get extraction API URL (prioritize session state, then env var, then hardcoded default)
def get_extraction_api_url() -> str:
    if "extraction_api_url" in st.session_state and st.session_state.extraction_api_url:
        return st.session_state.extraction_api_url
    env_url = os.environ.get("EXTRACTION_API_URL", "http://127.0.0.1:8000/extract") # Default to local if not set
    st.session_state.extraction_api_url = env_url # Store in session state
    return env_url

# Function to get API key for the extraction API (prioritize session state, then env var, then hardcoded default)
def get_api_key() -> str:
    if "api_key" in st.session_state and st.session_state.api_key:
        return st.session_state.api_key
    env_key = os.environ.get("API_KEY", "") # Refers to the key needed BY the extraction API
    if env_key:
        st.session_state.api_key = env_key # Store in session state
        return env_key
    # Avoid hardcoding secrets, return empty if not found
    st.session_state.api_key = ""
    return ""


# --- Utils/Helpers Module (MODIFIED for Detailed Structure) ---

def safe_get(data: Optional[Dict], keys: List[str], default: Any = 0.0) -> Any:
    """Safely get a nested value from a dictionary, handling None checks."""
    if data is None:
        # logger.debug(f"safe_get: Input data is None, returning default {default} for keys {keys}")
        return default
    temp = data
    for i, key in enumerate(keys):
        if isinstance(temp, dict) and key in temp:
            temp = temp.get(key)
            if temp is None: # Handle explicit None values safely
                 # Allow None for reserves, otherwise return default
                 is_reserve = keys[-1] in ['replacement_reserves', 'tenant_improvements'] and i == len(keys) - 1
                 if is_reserve:
                      # logger.debug(f"safe_get: Found None for reserve key {keys}, returning None")
                      return None
                 else:
                      # logger.debug(f"safe_get: Found None at key '{key}' in path {keys}, returning default {default}")
                      return default
        else:
            # logger.debug(f"safe_get: Key path not found: {keys} at key '{key}'. Data: {str(temp)[:100]}")
            return default

    # Final check if value is None after traversal (and not a reserve)
    is_reserve_final = keys[-1] in ['replacement_reserves', 'tenant_improvements']
    if temp is None and not is_reserve_final:
         # logger.debug(f"safe_get: Final value is None for non-reserve key {keys}, returning default {default}")
         return default

    # Handle potential NaN values from pandas/numpy if they sneak through
    if isinstance(temp, float) and math.isnan(temp):
        logger.warning(f"safe_get: NaN value found for key path {keys}, returning default {default}")
        return default

    # logger.debug(f"safe_get: Found value {temp} for keys {keys}")
    return temp

# --- THIS FUNCTION IS REWRITTEN ---
def format_for_noi_comparison(backend_response: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Formats the detailed financial data structure received from the ENHANCED backend API
    into a dictionary containing key metrics and breakdowns for comparison.

    Args:
        backend_response: The full dictionary response from the extraction API,
                          containing the nested 'financials' structure.

    Returns:
        A dictionary with key calculated metrics (GPR, EGI, Vacancy, OpEx, NOI)
        and nested dictionaries for breakdowns. Returns empty structure on error/None input.
    """
    default_breakdown = {}
    default_structure = {
        "gpr": 0.0, "vacancy_loss": 0.0, "other_income": 0.0, "egi": 0.0,
        "opex": 0.0, "noi": 0.0, "reserves_repl": None, "reserves_ti": None, # Use None for reserves default
        "vacancy_breakdown": default_breakdown, "other_income_breakdown": default_breakdown,
        "opex_breakdown": default_breakdown, "loss_to_lease": 0.0 # Added loss_to_lease
    }

    if backend_response is None or 'financials' not in backend_response:
        logger.warning("Received None or invalid data structure for formatting.")
        return default_structure

    fin = backend_response['financials'] # Main financials object
    logger.debug(f"Formatting data for period: {backend_response.get('period', 'Unknown')}")

    # --- Extract main components using safe_get ---
    gpr = safe_get(fin, ['income_summary', 'gross_potential_rent'], 0.0)
    loss_to_lease = safe_get(fin, ['income_summary', 'loss_to_lease'], 0.0)
    vacancy_loss = safe_get(fin, ['income_summary', 'total_vacancy_credit_loss'], 0.0)
    other_income = safe_get(fin, ['income_summary', 'total_other_income'], 0.0)
    egi = safe_get(fin, ['income_summary', 'effective_gross_income'], 0.0)
    opex = safe_get(fin, ['operating_expenses', 'total_operating_expenses'], 0.0)
    noi = safe_get(fin, ['net_operating_income'], 0.0) # Directly from top level of financials

    # Reserves (handle potential None, default to None if not found)
    reserves_repl = safe_get(fin, ['reserves', 'replacement_reserves'], default=None)
    reserves_ti = safe_get(fin, ['reserves', 'tenant_improvements'], default=None)

    # --- Extract breakdowns ---
    vacancy_breakdown = safe_get(fin, ['income_summary', 'vacancy_credit_loss_details'], default={})
    other_income_breakdown = safe_get(fin, ['income_summary', 'other_income_details'], default={})
    opex_breakdown = safe_get(fin, ['operating_expenses', 'expense_details'], default={})

    # --- Perform basic consistency checks (optional, backend should handle major ones) ---
    calc_egi = gpr - vacancy_loss + other_income
    if abs(calc_egi - egi) > 1.0:
        logger.warning(f"Frontend Formatting Check: EGI mismatch. API: {egi:.2f}, Calc (GPR-Vac+OI): {calc_egi:.2f}")
    calc_noi = egi - opex
    if abs(calc_noi - noi) > 1.0:
        logger.warning(f"Frontend Formatting Check: NOI mismatch. API: {noi:.2f}, Calc (EGI-OpEx): {calc_noi:.2f}")

    # --- Assemble the final dictionary ---
    formatted_data = {
        "gpr": float(gpr),
        "loss_to_lease": float(loss_to_lease),
        "vacancy_loss": float(vacancy_loss),
        "other_income": float(other_income),
        "egi": float(egi),
        "opex": float(opex),
        "noi": float(noi),
        "reserves_repl": reserves_repl, # Keep as None if applicable
        "reserves_ti": reserves_ti,     # Keep as None if applicable
        # Include breakdowns (ensure they are dicts)
        "vacancy_breakdown": vacancy_breakdown if isinstance(vacancy_breakdown, dict) else default_breakdown,
        "other_income_breakdown": other_income_breakdown if isinstance(other_income_breakdown, dict) else default_breakdown,
        "opex_breakdown": opex_breakdown if isinstance(opex_breakdown, dict) else default_breakdown
    }
    # logger.debug(f"Formatted data structure: {formatted_data}")
    return formatted_data
# --- END OF REWRITE ---


# --- NOI Calculations Module (MODIFIED for Detailed Structure) ---

# --- THIS FUNCTION IS REWRITTEN ---
def calculate_noi_comparisons(consolidated_data: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Calculate detailed NOI comparisons based on the enhanced consolidated data structure.

    Args:
        consolidated_data: Dictionary containing the formatted detailed financial data
                           for 'current_month', 'prior_month', 'budget', 'prior_year'.
                           Values should be the output of format_for_noi_comparison.

    Returns:
        A dictionary containing comparison results for various metrics (GPR, EGI, Vacancy, OpEx, NOI).
    """
    comparison_results = {}
    current_data = consolidated_data.get("current_month")

    # If no current data, cannot perform comparisons
    if not current_data:
        logger.warning("Cannot calculate comparisons: current_month data is missing.")
        return {}

    comparison_results["current"] = current_data # Store the whole formatted dict

    # --- Helper function for safe division & percentage calculation ---
    def safe_percent_change(current, previous):
        # Ensure both are numbers for calculation
        current_val = float(current) if isinstance(current, (int, float)) else 0.0
        previous_val = float(previous) if isinstance(previous, (int, float)) else 0.0

        if previous_val == 0:
            if current_val == 0: return 0.0 # No change from zero
            else: return None # Indicate infinite change / undefined percentage
        change = current_val - previous_val
        percentage = (change / previous_val) * 100
        return percentage

    # --- Define key metrics to compare ---
    metrics_to_compare = [
        "gpr", "loss_to_lease", "vacancy_loss", "other_income", "egi", "opex", "noi"
    ]

    # --- Calculate Month vs Prior Month Comparison ---
    prior_month_data = consolidated_data.get("prior_month")
    if prior_month_data:
        mom = {}
        for key in metrics_to_compare:
            current_val = current_data.get(key, 0.0)
            prior_val = prior_month_data.get(key, 0.0)
            mom[f"{key}_prior"] = prior_val
            mom[f"{key}_change"] = current_val - prior_val
            mom[f"{key}_percent_change"] = safe_percent_change(current_val, prior_val)
        comparison_results["month_vs_prior"] = mom
        logger.info("Calculated Month-over-Month comparisons.")

    # --- Calculate Actual vs Budget Comparison ---
    budget_data = consolidated_data.get("budget")
    if budget_data:
        avb = {}
        for key in metrics_to_compare:
            actual_val = current_data.get(key, 0.0)
            budget_val = budget_data.get(key, 0.0)
            avb[f"{key}_budget"] = budget_val
            avb[f"{key}_variance"] = actual_val - budget_val
            # Note: Interpretation of % variance (favorable/unfavorable) depends on the metric (e.g., lower expense variance is good)
            avb[f"{key}_percent_variance"] = safe_percent_change(actual_val, budget_val)
        comparison_results["actual_vs_budget"] = avb
        logger.info("Calculated Actual vs Budget comparisons.")

    # --- Calculate Actual vs Prior Year Comparison ---
    prior_year_data = consolidated_data.get("prior_year")
    if prior_year_data:
        yoy = {}
        for key in metrics_to_compare:
            current_val = current_data.get(key, 0.0)
            prior_val = prior_year_data.get(key, 0.0)
            yoy[f"{key}_prior_year"] = prior_val
            yoy[f"{key}_change"] = current_val - prior_val
            yoy[f"{key}_percent_change"] = safe_percent_change(current_val, prior_val)
        comparison_results["year_vs_year"] = yoy
        logger.info("Calculated Year-over-Year comparisons.")

    # logger.debug(f"Detailed comparison results structure: {json.dumps(comparison_results, default=str)}")
    return comparison_results
# --- END OF REWRITE ---


# --- AI Insights GPT Module (MODIFIED for Detailed Structure) ---

# --- THIS FUNCTION IS REWRITTEN ---
def format_detailed_comparison_results_for_prompt(comparison_results: Dict[str, Any]) -> str:
    """
    Formats the DETAILED comparison results into a string for the GPT prompt.

    Args:
        comparison_results: Results from calculate_noi_comparisons() using detailed data.

    Returns:
        Formatted string with detailed comparison results.
    """
    formatted_text = ""
    current = comparison_results.get("current")

    def fmt_val(value):
        """Formats numbers as currency, handles None/NaN."""
        if value is None or (isinstance(value, float) and math.isnan(value)): return "N/A"
        return f"${float(value):,.2f}"

    def fmt_pct(value):
        """Formats percentage, handles None/NaN."""
        if value is None or (isinstance(value, float) and math.isnan(value)): return "N/A"
        return f"{float(value):.1f}%"

    # --- Current Period ---
    if current:
        formatted_text += "CURRENT PERIOD DATA:\n"
        formatted_text += f"- Gross Potential Rent (GPR): {fmt_val(current.get('gpr'))}\n"
        formatted_text += f"- Loss to Lease:              {fmt_val(current.get('loss_to_lease'))}\n"
        formatted_text += f"- Vacancy & Credit Loss:    {fmt_val(current.get('vacancy_loss'))}\n"
        formatted_text += f"- Other Income:               {fmt_val(current.get('other_income'))}\n"
        formatted_text += f"- Effective Gross Income (EGI): {fmt_val(current.get('egi'))}\n"
        formatted_text += f"- Total Operating Expenses:   {fmt_val(current.get('opex'))}\n"
        formatted_text += f"- Net Operating Income (NOI): {fmt_val(current.get('noi'))}\n\n"

    # --- Comparison Table Helper ---
    def generate_comparison_table(comp_data, comp_label, prior_key_suffix, change_key, percent_key):
        if not comp_data or not current: return ""
        table = f"COMPARISON: CURRENT vs {comp_label.upper()}\n"
        table += f"{'Metric':<18} | {'Current':<15} | {comp_label:<15} | {'Change ($)':<15} | {'Change (%)':<10}\n"
        table += f"{'-'*18}-+-{'-'*15}-+-{'-'*15}-+-{'-'*15}-+-{'-'*10}\n"
        metrics = ["GPR", "Vacancy Loss", "Other Income", "EGI", "Total OpEx", "NOI"]
        data_keys = ["gpr", "vacancy_loss", "other_income", "egi", "opex", "noi"]
        for key, name in zip(data_keys, metrics):
            curr_val = current.get(key, 0.0)
            prior_val = comp_data.get(f"{key}_{prior_key_suffix}", 0.0)
            change_val = comp_data.get(f"{key}_{change_key}", 0.0)
            pct_val = comp_data.get(f"{key}_{percent_key}") # Can be None
            table += f"{name:<18} | {fmt_val(curr_val):<15} | {fmt_val(prior_val):<15} | {fmt_val(change_val):<15} | {fmt_pct(pct_val):<10}\n"
        return table + "\n"

    # --- Generate Tables ---
    formatted_text += generate_comparison_table(
        comparison_results.get("actual_vs_budget"), "Budget", "budget", "variance", "percent_variance"
    )
    formatted_text += generate_comparison_table(
        comparison_results.get("year_vs_year"), "Prior Year", "prior_year", "change", "percent_change"
    )
    formatted_text += generate_comparison_table(
        comparison_results.get("month_vs_prior"), "Prior Month", "prior", "change", "percent_change"
    )

    return formatted_text.strip()
# --- END OF REWRITE ---

# --- generate_insights_with_gpt and parse_gpt_response remain the same structurally ---
# They now receive richer data from the formatting function above.
def generate_insights_with_gpt(comparison_results: Dict[str, Any], property_name: str = "") -> Dict[str, Any]:
    """Generates insights using GPT based on DETAILED comparison results."""
    logger.info(f"Generating detailed insights with GPT for property: {property_name}")
    api_key = get_openai_api_key()
    if not api_key or len(api_key) < 10:
         logger.error("Invalid or missing OpenAI API key. Cannot generate insights.")
         st.error("OpenAI API Key is not configured correctly in Settings.")
         return {"summary": "Error: OpenAI API key not configured.", "performance": [], "recommendations": []}
    logger.info(f"Using OpenAI API key: {'*' * (len(api_key) - 5)}{api_key[-5:]}")
    client = OpenAI(api_key=api_key)
    formatted_results = format_detailed_comparison_results_for_prompt(comparison_results) # Use the detailed formatter
    prompt = f"""
    You are a senior real estate financial analyst providing insights on property performance based on detailed Net Operating Income (NOI) variance analysis.

    Property Name: {property_name or "This Property"}

    Analyze the following financial comparison data:
    ---
    {formatted_results}
    ---

    Based *only* on the data provided, please provide:
    1.  **Executive Summary:** A concise overview of the property's financial performance, highlighting the main drivers of NOI changes compared to budget and prior periods. Mention key metrics like EGI, Vacancy, and OpEx trends.
    2.  **Key Performance Insights (3-5 bullet points):** Specific, data-driven observations about significant variances or trends in revenue components (GPR, Vacancy, Other Income), operating expenses (major categories), and overall NOI. Quantify insights where possible (e.g., "Vacancy loss decreased by X%, contributing Y to NOI improvement").
    3.  **Actionable Recommendations (3-5 bullet points):** Concrete suggestions based *directly* on the observed variances to improve NOI. Focus on areas like reducing specific expenses, improving rent collection/reducing vacancy, increasing other income, or investigating budget variances.

    Format your response clearly with the numbered sections and bullet points as requested. Be professional and objective.
    """
    logger.info(f"Sending detailed prompt to GPT API (length: {len(prompt)} chars)...")
    try:
        response = client.chat.completions.create(
            model="gpt-4", # Or "gpt-4-turbo"
            messages=[
                {"role": "system", "content": "You are a senior real estate financial analyst specializing in detailed NOI variance analysis and reporting."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1200
        )
        response_content = response.choices[0].message.content
        logger.info(f"Received response from GPT API (length: {len(response_content)} chars).")
        insights = parse_gpt_response(response_content) # Reuse existing parser
        return insights
    except Exception as e:
        logger.error(f"Error generating insights with GPT: {str(e)}", exc_info=True)
        st.error(f"Error generating AI insights: {e}")
        return {"summary": f"Error generating insights: {str(e)}.", "performance": [], "recommendations": []}

def parse_gpt_response(content: str) -> Dict[str, Any]:
    """
    Parse the GPT response (potentially markdown formatted) into summary, performance, and recommendations.
    """
    logger.info(f"Parsing GPT response (first 100 chars): {content[:100]}...")
    insights = {"summary": "", "performance": [], "recommendations": []}
    current_section = None

    # Normalize line endings and split into lines
    lines = content.replace('\r\n', '\n').replace('\r', '\n').split('\n')

    for line in lines:
        stripped_line = line.strip()

        # Detect section headers (more flexible)
        if re.match(r"^\s*1\.\s*\*\*Executive Summary\*\*|Executive Summary:|SUMMARY", stripped_line, re.IGNORECASE):
            current_section = "summary"
            # Try to capture content on the same line after the header
            header_parts = re.split(r":", stripped_line, maxsplit=1)
            if len(header_parts) > 1 and header_parts[1].strip():
                 insights["summary"] += header_parts[1].strip() + " "
            continue
        elif re.match(r"^\s*2\.\s*\*\*Key Performance Insights\*\*|Key Performance Insights:|PERFORMANCE INSIGHTS", stripped_line, re.IGNORECASE):
            current_section = "performance"
            continue
        elif re.match(r"^\s*3\.\s*\*\*Actionable Recommendations\*\*|Actionable Recommendations:|RECOMMENDATIONS", stripped_line, re.IGNORECASE):
            current_section = "recommendations"
            continue

        # Skip empty lines or section dividers
        if not stripped_line or stripped_line in ['---', '***']:
            continue

        # Append content to the current section
        if current_section == "summary":
            insights["summary"] += stripped_line + " "
        elif current_section == "performance":
            # Handle bullet points (-, *, •, or digit.)
            match = re.match(r"^\s*[-*•\d]+\.?\s*(.*)", stripped_line)
            if match and match.group(1):
                insights["performance"].append(match.group(1).strip())
            elif insights["performance"]: # Append to last point if not a new bullet
                 insights["performance"][-1] += " " + stripped_line
            else: # If first line isn't a bullet, add it anyway
                 insights["performance"].append(stripped_line)

        elif current_section == "recommendations":
            match = re.match(r"^\s*[-*•\d]+\.?\s*(.*)", stripped_line)
            if match and match.group(1):
                insights["recommendations"].append(match.group(1).strip())
            elif insights["recommendations"]: # Append to last point
                 insights["recommendations"][-1] += " " + stripped_line
            else: # If first line isn't a bullet
                 insights["recommendations"].append(stripped_line)

    # Clean up whitespace and handle empty sections
    insights["summary"] = insights["summary"].strip()
    if not insights["summary"]: insights["summary"] = "Summary could not be parsed."
    if not insights["performance"]: insights["performance"] = ["No specific performance insights parsed."]
    if not insights["recommendations"]: insights["recommendations"] = ["No specific recommendations parsed."]

    logger.info(f"Parsed insights: Summary len={len(insights['summary'])}, Perf items={len(insights['performance'])}, Rec items={len(insights['recommendations'])}")
    return insights

# --- Insights Display Module (Unchanged - displays richer content now) ---
def display_insights(insights: Dict[str, Any], property_name: str = ""):
    """Displays GPT-generated insights in the Streamlit UI."""
    logger.info(f"Displaying detailed insights for property: {property_name}")
    with st.container():
        st.markdown("""<style>.insights-container{background-color:#f0f8ff;padding:20px;border-radius:10px;border-left:6px solid #1E90FF;margin-bottom:25px;box-shadow:2px 2px 5px rgba(0,0,0,0.1);}.insights-container h2{color:#1E90FF;margin-bottom:15px;}.insights-container h3{color:#4682B4;margin-top:15px;margin-bottom:5px;}</style>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="insights-container"><h2>AI-Generated Financial Insights</h2>""", unsafe_allow_html=True)
        st.subheader("Executive Summary")
        if insights and "summary" in insights and insights["summary"]: st.markdown(insights["summary"])
        else: st.info("No executive summary available.")
        st.subheader("Performance Analysis")
        if insights and "performance" in insights and insights["performance"]:
            for point in insights["performance"]: st.markdown(f"• {point}")
        else: st.info("No performance analysis available.")
        st.subheader("Recommendations")
        if insights and "recommendations" in insights and insights["recommendations"]:
            for rec in insights["recommendations"]: st.markdown(f"• {rec}")
        else: st.info("No recommendations available.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.caption("These insights were generated by AI based on the detailed financial data provided.")

# --- AI Extraction Module (MODIFIED - Calls enhanced backend) ---
def extract_noi_data(file: Any, document_type_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Extract DETAILED NOI data using the ENHANCED extraction API."""
    api_url = get_extraction_api_url()
    api_key = get_api_key() # Get key from session state/env var
    if not api_key or len(api_key) < 5:
         st.error("Extraction API Key is not configured correctly in settings.")
         logger.error("Missing or invalid extraction API key.")
         return None
    logger.info(f"Extracting detailed data from {file.name} using API: {api_url}")
    logger.info(f"Document type hint provided: {document_type_hint}")
    progress_bar = None
    status_text = st.empty() # Create placeholder for status text

    try:
        files_payload = {"file": (file.name, file.getvalue(), file.type)}
        data_payload = {}
        if document_type_hint: data_payload['document_type'] = document_type_hint
        headers = {"x-api-key": api_key}

        progress_bar = st.progress(0) # Initialize progress bar here
        status_text.text(f"🚀 Sending {file.name} for detailed extraction...")

        with st.spinner(f"Extracting detailed data from {file.name}..."):
            progress_bar.progress(30)
            logger.info(f"Sending POST to {api_url} with headers: {list(headers.keys())}, data_payload: {data_payload.keys()}")
            response = requests.post(api_url, files=files_payload, data=data_payload, headers=headers, timeout=120)
            progress_bar.progress(70)
            status_text.text("Processing API response...")

        progress_bar.progress(90)
        status_text.text("Finalizing extraction...")

        if response.status_code == 200:
            progress_bar.progress(100)
            status_text.success(f"✅ Detailed extraction complete for {file.name}!")
            time.sleep(1); status_text.empty(); progress_bar.empty()
            result = response.json()
            logger.info(f"Successfully extracted detailed data from {file.name}")
            if 'validation_warnings' in result and result['validation_warnings']:
                 logger.warning(f"Backend validation warnings for {file.name}: {result['validation_warnings']}")
                 st.warning(f"Data validation warnings for {file.name}: {'; '.join(result['validation_warnings'])}")
            return result
        else:
            logger.error(f"API error ({file.name}): {response.status_code} - {response.text}")
            error_detail = response.text
            try: error_detail = response.json().get("detail", response.text)
            except json.JSONDecodeError: pass
            st.error(f"API Error ({response.status_code}) for {file.name}: {error_detail}")
            status_text.error(f"❌ Extraction failed for {file.name}. Status: {response.status_code}");
            if progress_bar: progress_bar.empty()
            return None
    except requests.exceptions.Timeout:
        logger.error(f"Request timed out processing {file.name}"); st.error(f"Request timed out processing {file.name}."); status_text.error(f"❌ Timeout for {file.name}.")
        if progress_bar: progress_bar.empty()
        return None
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error to {api_url}"); st.error(f"Connection Error: Could not connect to API at {api_url}."); status_text.error(f"❌ Connection error for {file.name}.")
        if progress_bar: progress_bar.empty()
        return None
    except Exception as e:
        logger.error(f"Error extracting data ({file.name}): {str(e)}", exc_info=True); st.error(f"Unexpected error during extraction for {file.name}: {str(e)}"); status_text.error(f"❌ Error for {file.name}.")
        if progress_bar: progress_bar.empty()
        return None

# --- THIS FUNCTION IS REWRITTEN ---
def process_all_documents() -> Dict[str, Any]:
    """
    Process all uploaded documents using the enhanced extraction API and new formatter.

    Returns:
        Consolidated data dictionary with the DETAILED structure for each document type.
    """
    consolidated_data = {"current_month": None, "prior_month": None, "budget": None, "prior_year": None}
    processing_successful = True # Assume success initially

    # Define processing order and keys
    processing_map = {
        "current_month": ("Current Month", st.session_state.current_month_actuals, "current_month_actuals"),
        "prior_month": ("Prior Month", st.session_state.prior_month_actuals, "prior_month_actuals"),
        "budget": ("Budget", st.session_state.current_month_budget, "current_month_budget"),
        "prior_year": ("Prior Year", st.session_state.prior_year_actuals, "prior_year_actuals")
    }

    # Process each file type
    for key, (label, file_obj, type_hint) in processing_map.items():
        if file_obj:
            st.write(f"Processing {label}: {file_obj.name}...")
            result = extract_noi_data(file_obj, type_hint) # Pass the type hint
            if result:
                formatted_data = format_for_noi_comparison(result) # Use the new formatter
                consolidated_data[key] = formatted_data
                # Check if essential data was formatted (basic check)
                if key == "current_month" and (not formatted_data or formatted_data.get("noi") is None):
                     st.error(f"Failed to format essential data for Current Month: {file_obj.name}")
                     processing_successful = False
                else:
                     st.success(f"Processed {label}: {file_obj.name}")
            else:
                 st.error(f"Failed to process {label}: {file_obj.name}")
                 if key == "current_month": # Current month is essential
                      processing_successful = False
                 # Allow continuing even if optional files fail, but log it
                 logger.warning(f"Processing failed for optional file type '{key}', file: {file_obj.name}")
        else:
             logger.info(f"No file uploaded for {label}, skipping.")


    st.session_state.consolidated_data = consolidated_data
    st.session_state.processing_completed = processing_successful # Store overall status

    return consolidated_data
# --- END OF REWRITE ---


# --- Display Module (MODIFIED for Detailed Structure) ---

# --- THIS FUNCTION IS REWRITTEN ---
def display_noi_comparisons(comparison_results: Dict[str, Any]):
    """
    Display DETAILED NOI comparisons in the Streamlit app.

    Args:
        comparison_results: Results from calculate_noi_comparisons() using detailed data.
    """

    current_data = comparison_results.get("current")
    if not current_data:
        st.warning("No current period data available to display comparisons.")
        return

    st.header("Financial Performance Overview")
    st.markdown("Key metrics for the current period.")

    # --- Key Metrics Section ---
    col1, col2, col3, col4 = st.columns(4)
    current_gpr = current_data.get('gpr', 0.0)
    current_egi = current_data.get('egi', 0.0)
    current_vacancy = current_data.get('vacancy_loss', 0.0)
    current_opex = current_data.get('opex', 0.0)
    current_noi = current_data.get('noi', 0.0)

    with col1:
        st.metric("Eff. Gross Income (EGI)", f"${current_egi:,.2f}")
    with col2:
        vacancy_rate = (current_vacancy / current_gpr * 100) if current_gpr else 0
        st.metric("Vacancy & Credit Loss %", f"{vacancy_rate:.1f}%", f"-${current_vacancy:,.2f}", delta_color="inverse")
    with col3:
        opex_ratio = (current_opex / current_egi * 100) if current_egi else 0
        st.metric("Operating Expense Ratio", f"{opex_ratio:.1f}%", f"${current_opex:,.2f}", delta_color="inverse")
    with col4:
        st.metric("Net Operating Income (NOI)", f"${current_noi:,.2f}")

    st.markdown("---")

    # --- Comparison Tabs ---
    st.header("Comparative Analysis")
    tab_titles = ["Current vs Budget", "Current vs Prior Year", "Current vs Prior Month"]
    # Filter out tabs for which comparison data doesn't exist
    available_tabs = []
    if "actual_vs_budget" in comparison_results: available_tabs.append("Current vs Budget")
    if "year_vs_year" in comparison_results: available_tabs.append("Current vs Prior Year")
    if "month_vs_prior" in comparison_results: available_tabs.append("Current vs Prior Month")

    if not available_tabs:
        st.info("No comparison data available (e.g., no Budget or Prior Period files processed).")
        return

    tabs = st.tabs(available_tabs)
    tab_map = {
        "Current vs Budget": ("actual_vs_budget", "budget", "Budget", "variance", "percent_variance"),
        "Current vs Prior Year": ("year_vs_year", "prior_year", "Prior Year", "change", "percent_change"),
        "Current vs Prior Month": ("month_vs_prior", "prior", "Prior Month", "change", "percent_change")
    }

    # Helper to display comparison metrics
    def display_comparison_tab(tab_key, prior_key_suffix, comp_label, change_key, percent_key):
        tab_data = comparison_results.get(tab_key)
        if not tab_data:
            st.info(f"No {comp_label} data available for comparison.")
            return

        st.subheader(f"Current vs {comp_label}")
        metrics_display = ["GPR", "Vacancy Loss", "Other Income", "EGI", "Total OpEx", "NOI"]
        data_keys = ["gpr", "vacancy_loss", "other_income", "egi", "opex", "noi"]

        df_data = []
        for key, name in zip(data_keys, metrics_display):
            current_val = current_data.get(key, 0.0)
            prior_val = tab_data.get(f"{key}_{prior_key_suffix}", 0.0)
            change_val = tab_data.get(f"{key}_{change_key}", 0.0)
            percent_change_val = tab_data.get(f"{key}_{percent_key}") # Can be None

            df_data.append({
                "Metric": name,
                "Current": current_val,
                comp_label: prior_val,
                "Change ($)": change_val,
                "Change (%)": percent_change_val if percent_change_val is not None else np.nan # Use NaN for formatting N/A
            })

        df_display = pd.DataFrame(df_data)
        st.dataframe(
            df_display.style.format({
                "Current": "${:,.2f}",
                comp_label: "${:,.2f}",
                "Change ($)": "{:,.2f}",
                "Change (%)": "{:.1f}%"
            }, na_rep="N/A"), # Display NaN as N/A
            use_container_width=True
        )

        # --- Add Charts for the Tab ---
        st.subheader(f"Visual Comparison: Current vs {comp_label}")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Bar chart comparing EGI and NOI
            fig_bar = go.Figure(data=[
                go.Bar(name='Current', x=['EGI', 'NOI'], y=[current_data.get('egi',0), current_data.get('noi',0)], marker_color='#1f77b4'), # Blue
                go.Bar(name=comp_label, x=['EGI', 'NOI'], y=[tab_data.get(f"egi_{prior_key_suffix}", 0), tab_data.get(f"noi_{prior_key_suffix}", 0)], marker_color='#ff7f0e') # Orange
            ])
            fig_bar.update_layout(barmode='group', title=f'EGI & NOI Comparison', yaxis_title="Amount ($)", legend_title_text='Period')
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_col2:
             # Bar chart comparing Vacancy Loss and OpEx
             fig_bar_loss = go.Figure(data=[
                 go.Bar(name='Current', x=['Vacancy Loss', 'Total OpEx'], y=[current_data.get('vacancy_loss',0), current_data.get('opex',0)], marker_color='#d62728'), # Red
                 go.Bar(name=comp_label, x=['Vacancy Loss', 'Total OpEx'], y=[tab_data.get(f"vacancy_loss_{prior_key_suffix}", 0), tab_data.get(f"opex_{prior_key_suffix}", 0)], marker_color='#9467bd') # Purple
             ])
             fig_bar_loss.update_layout(barmode='group', title=f'Vacancy Loss & OpEx Comparison', yaxis_title="Amount ($)", legend_title_text='Period')
             st.plotly_chart(fig_bar_loss, use_container_width=True)

        # --- Optional: Display Breakdowns ---
        with st.expander(f"Show Detailed Breakdown vs {comp_label}"):
            st.write("_(Breakdown comparisons coming soon...)_")
            # Example: Compare OpEx breakdown
            # current_opex_detail = current_data.get("opex_breakdown", {})
            # prior_opex_detail = tab_data.get(f"opex_{prior_key_suffix}_breakdown", {}) # Need to add breakdowns to comparison results
            # if current_opex_detail: # Check if breakdown exists
            #     st.write("Operating Expense Breakdown Comparison:")
            #     # Create a DataFrame or display side-by-side...
            # else:
            #     st.write("Detailed OpEx breakdown not available.")


    # Populate Tabs dynamically
    for i, tab_title in enumerate(available_tabs):
        with tabs[i]:
            tab_key, prior_key_suffix, comp_label, change_key, percent_key = tab_map[tab_title]
            display_comparison_tab(tab_key, prior_key_suffix, comp_label, change_key, percent_key)

    st.markdown("---")
# --- END OF REWRITE ---


# --- Main Application Logic (MODIFIED - Uses new functions) ---

# --- show_upload_page, show_results_page, show_settings_page ---
# These functions are largely the same as the previous 'Enhanced' version,
# but they now call the updated process_all_documents, display_noi_comparisons,
# and generate_insights_with_gpt functions which handle the detailed data.
# Ensure keys used for session_state access are consistent.

def show_upload_page():
    """Show the upload page with clearly labeled file upload slots."""
    st.title("Enhanced NOI Calculation Tool")
    st.caption(f"Today is {datetime.now().strftime('%A, %B %d, %Y')}")

    st.text_input("Property Name", key="property_name", help="Enter property name")
    st.checkbox("Use sample data instead", key="using_sample_data", help="Use sample data (Note: simplified structure)")

    if not st.session_state.using_sample_data:
        st.header("📂 Upload Financial Statements")
        st.write("Upload documents for the periods you want to analyze.")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Current Month")
            st.file_uploader("1. Upload Current Month Income Statement (Actuals)", type=["csv", "xlsx", "xls", "pdf", "txt"], key="current_month_actuals", accept_multiple_files=False, help="Required: Current month actuals")
            if st.session_state.current_month_actuals: st.success(f"Loaded: {st.session_state.current_month_actuals.name}")
            st.subheader("Prior Month")
            st.file_uploader("2. Upload Prior Month Income Statement (Actuals)", type=["csv", "xlsx", "xls", "pdf", "txt"], key="prior_month_actuals", accept_multiple_files=False, help="Optional: Prior month actuals")
            if st.session_state.prior_month_actuals: st.success(f"Loaded: {st.session_state.prior_month_actuals.name}")
        with col2:
            st.subheader("Budget")
            st.file_uploader("3. Upload Current Month Budget", type=["csv", "xlsx", "xls", "pdf", "txt"], key="current_month_budget", accept_multiple_files=False, help="Optional: Budget figures")
            if st.session_state.current_month_budget: st.success(f"Loaded: {st.session_state.current_month_budget.name}")
            st.subheader("Prior Year")
            st.file_uploader("4. Upload Prior Year Income Statement (Actuals)", type=["csv", "xlsx", "xls", "pdf", "txt"], key="prior_year_actuals", accept_multiple_files=False, help="Optional: Prior year actuals")
            if st.session_state.prior_year_actuals: st.success(f"Loaded: {st.session_state.prior_year_actuals.name}")

        st.subheader("📊 Process & Analyze")
        if st.session_state.current_month_actuals is not None:
            if st.button("Process Documents and Calculate Detailed NOI"):
                st.session_state.consolidated_data = {}; st.session_state.comparison_results = {}; st.session_state.insights = None; st.session_state.processing_completed = False
                with st.spinner("Processing documents and calculating detailed NOI... Please wait."):
                    consolidated_data = process_all_documents() # Calls updated function
                    if not st.session_state.processing_completed: st.error("Processing failed. Cannot analyze."); return
                    if not consolidated_data.get("current_month"): st.error("Current month data missing after processing."); return
                    comparison_results = calculate_noi_comparisons(consolidated_data) # Calls updated function
                    st.session_state.comparison_results = comparison_results
                    try:
                        st.write("Generating AI Insights..."); insights = generate_insights_with_gpt(comparison_results, st.session_state.property_name) # Calls updated function
                        st.session_state.insights = insights; st.success("AI insights generated!")
                    except Exception as e: st.warning(f"Could not generate insights: {e}"); logger.error(f"Insight generation failed: {e}", exc_info=True); st.session_state.insights = None
                    st.session_state.page = "Results"; st.rerun()
        else: st.warning("Please upload at least the Current Month Income Statement.")
    else: # Sample Data Logic
        st.info("Using sample data for demonstration purposes.")
        st.warning("Note: Sample data uses a simplified structure.")
        if st.button("Process Sample Data"):
             with st.spinner("Processing sample data..."):
                # --- Use the DETAILED SAMPLE DATA defined earlier ---
                sample_current = {"gpr": 160000.0, "vacancy_loss": 8000.0, "other_income": 2000.0, "egi": 154000.0, "opex": 90000.0, "noi": 64000.0, "reserves_repl": 1000.0, "reserves_ti": 500.0, "vacancy_breakdown": {"physical_vacancy_loss": 6000, "concessions_free_rent": 1500, "bad_debt": 500}, "other_income_breakdown": {"recoveries": 1000, "parking_income": 500, "laundry_income": 300, "other_misc_income": 200, "application_fees": 0}, "opex_breakdown": {"property_taxes": 25000, "insurance": 5000, "property_management_fees": 8000, "repairs_maintenance": 12000, "utilities": 15000, "payroll": 10000, "admin_office_costs": 5000, "marketing_advertising": 3000, "other_opex": 7000}, "loss_to_lease": 0.0}
                sample_prior_month = {"gpr": 158000.0, "vacancy_loss": 9500.0, "other_income": 1800.0, "egi": 150300.0, "opex": 88000.0, "noi": 62300.0, "reserves_repl": 1000.0, "reserves_ti": 500.0, "vacancy_breakdown": {}, "other_income_breakdown": {}, "opex_breakdown": {}, "loss_to_lease": 0.0}
                sample_budget = {"gpr": 165000.0, "vacancy_loss": 7500.0, "other_income": 2500.0, "egi": 160000.0, "opex": 92000.0, "noi": 68000.0, "reserves_repl": 1200.0, "reserves_ti": 600.0, "vacancy_breakdown": {}, "other_income_breakdown": {}, "opex_breakdown": {}, "loss_to_lease": 0.0}
                sample_prior_year = {"gpr": 150000.0, "vacancy_loss": 10000.0, "other_income": 1500.0, "egi": 141500.0, "opex": 85000.0, "noi": 56500.0, "reserves_repl": 800.0, "reserves_ti": 400.0, "vacancy_breakdown": {}, "other_income_breakdown": {}, "opex_breakdown": {}, "loss_to_lease": 0.0}
                consolidated_data = {"current_month": sample_current, "prior_month": sample_prior_month, "budget": sample_budget, "prior_year": sample_prior_year}
                st.session_state.consolidated_data = consolidated_data
                comparison_results = calculate_noi_comparisons(consolidated_data) # Calls updated function
                st.session_state.comparison_results = comparison_results
                try:
                    st.write("Generating AI Insights for sample data..."); insights = generate_insights_with_gpt(comparison_results, st.session_state.property_name or "Sample Property") # Calls updated function
                    st.session_state.insights = insights
                except Exception as e: st.warning(f"Could not generate insights: {e}"); logger.error(f"Sample insight failed: {e}", exc_info=True); st.session_state.insights = None
                st.session_state.page = "Results"; st.rerun()


def show_results_page():
    """Show the results page with DETAILED comparisons and insights."""
    st.title("Detailed NOI Analysis Results")
    prop_name = st.session_state.get("property_name")
    if prop_name: st.subheader(f"Property: {prop_name}")
    comparison_results = st.session_state.get("comparison_results")
    if comparison_results:
        display_noi_comparisons(comparison_results) # Calls updated display function
        insights = st.session_state.get("insights")
        if insights: display_insights(insights, prop_name) # Calls display function
        else: st.info("AI insights were not generated or are unavailable.")
    else:
        st.warning("No comparison results available. Please process documents first.")
        if st.button("Go to Upload Page"): st.session_state.page = "Upload"; st.rerun()
    st.markdown("---")
    if st.button("⬅️ Return to Upload Page"): st.session_state.page = "Upload"; st.rerun()

def show_settings_page():
    """Show the settings page with API configuration options."""
    st.title("⚙️ Settings")
    st.header("API Configuration")
    st.info("Configure the API endpoints and keys for the backend services.")
    st.text_input("Extraction API URL", key="extraction_api_url", help="URL of the backend data extraction API (e.g., https://your-render-url.onrender.com/extract)")
    st.text_input("Extraction API Key", key="api_key", type="password", help="API key for the backend data extraction API")
    st.text_input("OpenAI API Key", key="openai_api_key", type="password", help="API key for OpenAI (for insights)")
    st.subheader("API Status")
    if st.button("Check Extraction API Status"):
         api_url_to_check = st.session_state.extraction_api_url
         api_key_to_check = st.session_state.api_key
         if not api_url_to_check: st.error("Extraction API URL is not set.")
         else:
              health_url = api_url_to_check.replace("/extract", "/health").replace("/extract-batch", "/health") # Try to find health endpoint
              try:
                   headers = {"x-api-key": api_key_to_check} if api_key_to_check else {}
                   # Health check might not need key, but try with it if provided
                   response = requests.get(health_url, timeout=10, headers=headers)
                   if response.status_code == 200: st.success(f"✅ API Health Check Successful ({health_url}) - Status: {response.json().get('status', 'OK')}")
                   # Handle potential 401/403 if health check needs key but wasn't provided/correct
                   elif response.status_code in [401, 403]: st.warning(f"⚠️ API Health Check at {health_url} returned {response.status_code}. Check API Key?")
                   else: st.warning(f"⚠️ API Health Check returned status {response.status_code} at {health_url}")
              except requests.exceptions.RequestException as e: st.error(f"❌ Failed to connect to API Health Check at {health_url}: {e}")
    st.markdown("---")
    if st.button("⬅️ Return to Upload Page"): st.session_state.page = "Upload"; st.rerun()


# --- Main Application Entry Point ---
def main():
    """Main function to run the Streamlit app"""
    st.set_page_config(page_title="Detailed NOI Analyzer", page_icon="💰", layout="wide", initial_sidebar_state="expanded")

    # Initialize session state variables robustly
    defaults = {
        "page": "Upload", "property_name": "", "using_sample_data": False,
        "current_month_actuals": None, "prior_month_actuals": None,
        "current_month_budget": None, "prior_year_actuals": None,
        "consolidated_data": {}, "comparison_results": {}, "insights": None,
        "processing_completed": False,
        # API Configs - fetch using functions to prioritize session state
        "openai_api_key": get_openai_api_key(),
        "extraction_api_url": get_extraction_api_url(),
        "api_key": get_api_key()
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # --- Sidebar Navigation ---
    st.sidebar.title("📈 NOI Analyzer")
    st.sidebar.markdown("---")
    page_options = ["Upload", "Results", "Settings"]
    # Use radio buttons for clearer navigation state
    # Ensure the current page value is valid before setting index
    current_page_index = page_options.index(st.session_state.page) if st.session_state.page in page_options else 0
    selected_page = st.sidebar.radio("Navigation", page_options, index=current_page_index)
    # Update state only if changed to prevent unnecessary reruns
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun() # Use st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.info("Upload financial documents, view analysis results, or configure API settings.")

    # --- Page Display Logic ---
    if st.session_state.page == "Upload":
        show_upload_page()
    elif st.session_state.page == "Results":
        show_results_page()
    elif st.session_state.page == "Settings":
        show_settings_page()
    else: # Fallback
        st.session_state.page = "Upload"; st.rerun()


if __name__ == "__main__":
    main()
