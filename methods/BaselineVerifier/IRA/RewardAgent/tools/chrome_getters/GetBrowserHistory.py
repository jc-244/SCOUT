from smolagents import Tool
from typing import Any, Dict, Optional
import os


from desktop_env.evaluators.getters.chrome import get_history


class GetBrowserHistoryTool(Tool):
    
    name = "get_browser_history"
    description = (
        "Get browsing history from Chrome. This can help verify if specific websites "
        "were visited or if history was cleared. Optionally filter by URL or time range. "
        "Returns history information as a string."
    )
    inputs = {
        "url_pattern": {
            "description": "Optional: URL pattern to search for in history",
            "type": "string",
            "nullable": True,
        }
    }
    output_type = "string"
    
    def __init__(self, env: Any):
        self.env = env
        super().__init__()
    
    def forward(self, url_pattern: Optional[str] = None) -> str:
        return self.__call__(url_pattern)
    
    def __call__(self, url_pattern: Optional[str] = None) -> str:
        try:
            config = {
                "dest": "chrome_history_temp.db"
            }
            if url_pattern:
                config["url_pattern"] = url_pattern
                
            result = get_history(self.env, config)
            
            if result is None:
                return "Error: Failed to get browser history"
            
            return str(result)
        except Exception as e:
            return f"Error getting browser history: {str(e)}"
    
    def to_code_prompt(self) -> str:
        return (
            "def get_browser_history(url_pattern: str = None) -> str:\n"
            "    '''Get browsing history from Chrome, optionally filtered by URL pattern.'''\n"
        )
