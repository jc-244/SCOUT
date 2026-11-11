from smolagents import Tool
from typing import Any, Dict, Optional
import os


from desktop_env.evaluators.getters.chrome import get_cookie_data


class GetCookieDataTool(Tool):
    
    name = "get_cookie_data"
    description = (
        "Get cookie data from Chrome for a specific domain or URL. "
        "This can help verify if cookies were set correctly or if login sessions exist. "
        "Returns cookie information as a string."
    )
    inputs = {
        "domain": {
            "description": "Domain to get cookies for (e.g., 'google.com')",
            "type": "string",
        }
    }
    output_type = "string"
    
    def __init__(self, env: Any):
        self.env = env
        super().__init__()
    
    def forward(self, domain: str) -> str:
        return self.__call__(domain)
    
    def __call__(self, domain: str) -> str:
        try:
            config = {"domain": domain}
            result = get_cookie_data(self.env, config)
            
            if result is None:
                return f"Error: Failed to get cookie data for domain '{domain}'"
            
            return str(result)
        except Exception as e:
            return f"Error getting cookie data for '{domain}': {str(e)}"
    
    def to_code_prompt(self) -> str:
        return (
            "def get_cookie_data(domain: str) -> str:\n"
            "    '''Get cookie data from Chrome for a specific domain.'''\n"
        )
