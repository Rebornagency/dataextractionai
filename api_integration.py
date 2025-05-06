"""
API Integration module for NOI Analyzer
Provides functions to interact with extraction API endpoints
"""

import os
import logging
import requests
from typing import Tuple, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('api_integration')

# Default API endpoint
DEFAULT_API_URL = os.environ.get('EXTRACTION_API_URL', 'http://localhost:8000/extract')

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.exceptions.ConnectionError, 
                                   requests.exceptions.Timeout,
                                   requests.exceptions.HTTPError))
)
def post_with_retry(url, files, headers=None, timeout=30):
    """
    Make a POST request with retry logic for transient errors
    
    Args:
        url: The API endpoint URL
        files: Files to upload
        headers: Optional request headers
        timeout: Request timeout in seconds
        
    Returns:
        Response object
        
    Raises:
        Exception: If all retry attempts fail
    """
    logger.info(f"Sending POST request to {url}")
    response = requests.post(url, files=files, headers=headers, timeout=timeout)
    response.raise_for_status()  # Raise exception for 4XX/5XX status codes
    return response

def call_extraction_api(file, api_url: Optional[str] = None, api_key: Optional[str] = None) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Call the extraction API with a file
    Enhanced with retry logic and improved error handling
    
    Args:
        file: File object to process
        api_url: Optional API URL (defaults to environment variable or localhost)
        api_key: Optional API key
        
    Returns:
        Tuple of (result_dict, error_message)
        If successful, error_message will be None
        If failed, result_dict will contain error information
    """
    url = api_url or DEFAULT_API_URL
    headers = {}
    
    if api_key:
        headers['X-API-Key'] = api_key
    
    try:
        logger.info(f"Sending {file.name} to extraction API at {url}")
        
        # Create files dict for multipart upload
        files = {
            'file': (file.name, file, file.type if hasattr(file, 'type') else 'application/octet-stream')
        }
        
        # Call API with retry logic
        response = post_with_retry(url, files=files, headers=headers)
        
        # Parse response
        result = response.json()
        
        # Check for error in the response
        if 'error' in result:
            logger.warning(f"API returned error: {result['error']}")
            return result, result['error']
        
        logger.info(f"Successfully extracted data from {file.name}")
        return result, None
        
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Connection error: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "status": "error"}, error_msg
        
    except requests.exceptions.Timeout as e:
        error_msg = f"Request timed out: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "status": "error"}, error_msg
        
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP error: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "status": "error"}, error_msg
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Request failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "status": "error"}, error_msg
        
    except ValueError as e:
        error_msg = f"Invalid response (JSON parsing failed): {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "status": "error"}, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg, "status": "error"}, error_msg 