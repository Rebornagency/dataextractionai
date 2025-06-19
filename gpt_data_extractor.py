import os
import logging
from typing import Dict, Any, List, Optional, Union
from openai import OpenAI
import json
import re
from ai_extraction.prompts import PROMPT_TEMPLATES, DEFAULT_PROMPT
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gpt_data_extractor')

class GPTDataExtractor:
    """
    Class for extracting financial data from documents using GPT-4
    Enhanced to extract detailed income components and application fees
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the data extractor

        Args:
            api_key: OpenAI API key (optional, can be set via environment variable)
        """
        # Set API key if provided, otherwise use environment variable
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            logger.warning("OpenAI API key not set. Please set OPENAI_API_KEY environment variable or provide it during initialization.")

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

    def extract_data(self, text_or_data: Union[str, Dict[str, Any]], document_type: str, period: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract financial data from document

        Args:
            text_or_data: Preprocessed text from the document or data dictionary
            document_type: Type of document (Actual Income Statement, Budget, Prior Year Actual)
            period: Time period covered by the document (optional)

        Returns:
            Dict containing extracted financial data
        """
        logger.info(f"Extracting data from {document_type} document for period {period}")

        # Extract text from input
        text = self._extract_text_from_input(text_or_data)
        if not text:
            logger.warning("No text content to extract data from")
            return {'error': 'No text content to extract data from'}

        # Truncate text if too long
        max_length = 12000  # Limit to avoid token limits
        if len(text) > max_length:
            # Take first and last parts
            first_part = text[:max_length//2]
            last_part = text[-max_length//2:]
            truncated_text = first_part + "\n...[content truncated]...\n" + last_part
            logger.info(f"Text truncated from {len(text)} to {len(truncated_text)} characters")
        else:
            truncated_text = text

        try:
            # Create prompt for GPT based on document type
            prompt = self._create_extraction_prompt(truncated_text, document_type, period)
            
            # Call GPT API with retry logic
            extraction_result = self._call_gpt_with_retry(prompt)
            
            # Extract raw lines from the document for auditing purposes
            raw_lines = self._extract_raw_lines(truncated_text, extraction_result)
            
            # Attach audit lines and metadata without altering the core JSON structure
            extraction_result["audit_lines"] = raw_lines
            extraction_result.setdefault("metadata", {})
            extraction_result["metadata"].update({
                "document_type": document_type,
                "period": period,
                "extraction_method": "gpt-4",
            })
            
            # Perform basic validation
            extraction_result = self._validate_extraction(extraction_result)
            
            return extraction_result
            
        except Exception as e:
            logger.error(f"Error in GPT data extraction: {str(e)}", exc_info=True)
            return {
                'error': f"Data extraction failed: {str(e)}",
                'metadata': {
                    'document_type': document_type,
                    'period': period
                }
            }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)),
        reraise=True
    )
    def _call_gpt_with_retry(self, prompt: str) -> Dict[str, Any]:
        """
        Call GPT API with retry logic for transient errors

        Args:
            prompt: The prompt to send to GPT

        Returns:
            Extracted data as dict
        
        Raises:
            Exception: If all retry attempts fail
        """
        try:
            logger.info("Calling GPT API...")
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial data extraction assistant that responds only in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            logger.info("GPT API call successful")
            return json.loads(response_text)
            
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {str(e)}. Retrying...")
            raise
        except openai.APITimeoutError as e:
            logger.warning(f"API timeout: {str(e)}. Retrying...")
            raise
        except openai.APIConnectionError as e:
            logger.warning(f"API connection error: {str(e)}. Retrying...")
            raise
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GPT response as JSON: {str(e)}")
            raise ValueError(f"Failed to parse GPT response: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in GPT API call: {str(e)}", exc_info=True)
            raise

    def _extract_text_from_input(self, text_or_data: Union[str, Dict[str, Any]]) -> str:
        """
        Extract text content from input which could be string or dictionary

        Args:
            text_or_data: Input data which could be string or dictionary

        Returns:
            Extracted text content
        """
        # Handle both string and dictionary input
        if isinstance(text_or_data, dict):
            # Try to extract text from common dictionary structures
            if 'combined_text' in text_or_data:
                return text_or_data['combined_text']
            elif 'text' in text_or_data:
                if isinstance(text_or_data['text'], str):
                    return text_or_data['text']
                elif isinstance(text_or_data['text'], list) and len(text_or_data['text']) > 0:
                    # Join text from multiple pages
                    page_contents = []
                    for page in text_or_data['text']:
                        if isinstance(page, dict) and 'content' in page and page['content']:
                            page_contents.append(page['content'])
                        elif isinstance(page, str): # Handle case where list contains strings
                            page_contents.append(page)
                    return "\n\n".join(page_contents)

            elif 'content' in text_or_data and isinstance(text_or_data['content'], str):
                return text_or_data['content']
            elif 'data' in text_or_data and isinstance(text_or_data['data'], str):
                return text_or_data['data']
            elif 'sheets' in text_or_data and isinstance(text_or_data['sheets'], list):
                 # Extract text from Excel sheets text representation
                 if 'text_representation' in text_or_data and isinstance(text_or_data['text_representation'], list):
                      return "\n\n".join(text_or_data['text_representation'])

            # If we couldn't extract text using known structures, convert the entire dictionary to a string
            logger.warning("Could not find specific text field in dictionary, using JSON string representation")
            try:
                # Limit the depth and length to avoid overly long strings
                return json.dumps(text_or_data, indent=2, default=str, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Error converting dict to JSON string: {e}")
                return str(text_or_data)[:2000] # Fallback to basic string conversion, limited length
        else:
            # Use the input directly if it's already a string
            if isinstance(text_or_data, str):
                return text_or_data
            else:
                # Convert to string if it's not a string
                logger.warning(f"Converted non-string input to string: {type(text_or_data)}")
                return str(text_or_data)

    def _create_extraction_prompt(self, text: str, document_type: str, period: Optional[str] = None) -> str:
        """
        Build a single, contract-compliant prompt string instructing the model to return
        ONLY the JSON defined by the NOI Analyzer schema.

        Args:
            text: The (possibly pre-processed) document text. Will be truncated to first 5k chars.
            document_type: Friendly name (e.g. "Actual Income Statement") coming from upstream.
            period: Optional period (YYYY-MM) supplied by caller.

        Returns:
            A fully-rendered prompt ready to be sent to the OpenAI chat completion endpoint.
        """
        # Truncate to keep token usage under control but still provide ample context
        text_sample = text[:5000]

        period_context = period if period else "Unknown"

        # NOTE: The JSON schema must match 100 % the expectations of the downstream
        # NOI Analyzer front-end.  Keep nesting and key order exactly.
        schema_block = (
            '{\n'
            '  "property_id": "string|null",\n'
            '  "period": "YYYY-MM",\n'
            '  "financials": {\n'
            '    "gross_potential_rent":           "number|null",\n'
            '    "vacancy_loss":                   "number|null",\n'
            '    "concessions":                    "number|null",\n'
            '    "bad_debt":                       "number|null",\n'
            '    "effective_gross_income":         "number|null",\n'
            '    "other_income": {\n'
            '      "total":                       "number|null",\n'
            '      "application_fees":            "number|0",\n'
            '      "parking":                     "number|0",\n'
            '      "laundry":                     "number|0",\n'
            '      "late_fees":                   "number|0",\n'
            '      "pet_fees":                    "number|0",\n'
            '      "storage_fees":                "number|0",\n'
            '      "amenity_fees":                "number|0",\n'
            '      "utility_reimbursements":      "number|0",\n'
            '      "cleaning_fees":               "number|0",\n'
            '      "cancellation_fees":           "number|0",\n'
            '      "miscellaneous":               "number|0",\n'
            '      "additional_items":            [ { "label": "string", "amount": "number" } ]\n'
            '    },\n'
            '    "operating_expenses": {\n'
            '      "total_operating_expenses":     "number|null",\n'
            '      "payroll":                     "number|0",\n'
            '      "administrative":              "number|0",\n'
            '      "marketing":                   "number|0",\n'
            '      "utilities":                   "number|0",\n'
            '      "repairs_and_maintenance":     "number|0",\n'
            '      "contract_services":           "number|0",\n'
            '      "make_ready":                  "number|0",\n'
            '      "turnover":                    "number|0",\n'
            '      "property_taxes":             "number|0",\n'
            '      "insurance":                  "number|0",\n'
            '      "management_fees":            "number|0"\n'
            '    },\n'
            '    "net_operating_income":           "number|null"\n'
            '  },\n'
            '  "metadata": {\n'
            '    "filename":        "string",\n'
            '    "document_type":   "string",\n'
            '    "period":          "YYYY-MM",\n'
            '    "extraction_time": "ISO-8601",\n'
            '    "property_id":     "string|null"\n'
            '  },\n'
            '  "validation_issues": [ "string" ],\n'
            '  "audit_lines":       [ "string" ]\n'
            '}'
        )

        prompt = f"""
You are NOI-GPT, a precision extraction engine for multifamily property financial statements.

TASK
1. Read the SOURCE TEXT and locate every field required by the JSON SCHEMA below.
2. Think through the math and sanity-check subtotals internally (do not show your work).
3. Respond with one and only one JSON object that strictly matches the schema. Use null when a value is missing. Strip currency symbols and thousands separators so every numeric field parses with float().
4. Do NOT wrap the JSON in markdown.

JSON SCHEMA
{schema_block}

CONTEXT
• Document category: {document_type}
• Period: {period_context}

SOURCE TEXT (truncated to 5 000 chars)
<<<
{text_sample}
>>>
"""
        return prompt

    def _extract_raw_lines(self, text: str, extraction_result: Dict[str, Any]) -> List[str]:
        """
        Extract raw text lines from the document that likely contain the extracted values
        This helps with auditing and debugging the extraction process

        Args:
            text: Document text
            extraction_result: Extracted data

        Returns:
            List of raw text lines relevant to the extraction
        """
        raw_lines = []
        
        # Find all dollar amounts and percentages in the text
        amount_regex = r'\$\s*[\d,]+(\.\d+)?|\d+\s*%|[\d,]+\.\d+|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b'
        
        # Get values from the extraction result
        extracted_values = []
        
        def extract_values(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, (int, float)) and value is not None:
                        extracted_values.append(str(value))
                    elif isinstance(value, (dict, list)):
                        extract_values(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_values(item)
        
        extract_values(extraction_result)
        
        # Convert values to strings without commas, dollar signs, etc. for matching
        extracted_values = [str(value).strip().replace('$', '').replace(',', '').replace('%', '') 
                           for value in extracted_values if value]
        
        # Split text into lines
        lines = text.split('\n')
        
        # Check each line
        for line in lines:
            # If line contains a dollar amount or percentage
            if re.search(amount_regex, line):
                # Check if any of the extracted values are in the line
                clean_line = line.replace('$', '').replace(',', '').replace('%', '')
                for value in extracted_values:
                    if value in clean_line:
                        raw_lines.append(line.strip())
                        break
            
            # Also add any line that looks like a financial header
            financial_terms = ['rent', 'income', 'expense', 'operating', 'vacancy', 
                              'gross', 'net', 'total', 'revenue', 'loss']
            if any(term in line.lower() for term in financial_terms) and len(line.strip()) > 5:
                raw_lines.append(line.strip())
        
        # Deduplicate and limit to a reasonable number
        raw_lines = list(dict.fromkeys(raw_lines))
        
        return raw_lines[:50]  # Limit to 50 lines

    def _validate_extraction(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform basic validation on the extracted data
        Flag potential issues for review

        Args:
            extraction_result: Extracted data

        Returns:
            Validated data with validation flags
        """
        result = extraction_result.copy()
        validation_issues = []
        
        financials = result.get("financials", {})
        
        # Check if NOI > GPR (invalid)
        noi = financials.get("net_operating_income")
        gpr = financials.get("gross_potential_rent")
        
        if noi is not None and gpr is not None and noi > gpr:
            validation_issues.append("NOI cannot exceed GPR")
        
        # Check if vacancy loss > GPR (invalid)
        vacancy = financials.get("vacancy_loss")
        if vacancy is not None and gpr is not None and vacancy > gpr:
            validation_issues.append("Vacancy loss cannot exceed GPR")
        
        # Add validation issues to result
        if validation_issues:
            result["validation_issues"] = validation_issues
            logger.warning(f"Validation issues found: {validation_issues}")
        
        return result
