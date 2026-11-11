from smolagents import Tool
from typing import Any, Dict, Optional
import os


from desktop_env.evaluators.getters.chrome import get_page_info


class GetPageInfoTool(Tool):
    
    name = "get_page_info"
    description = (
        "Get detailed information about the current page in Chrome, such as "
        "page title, URL, content, etc. This can help verify if the page "
        "loaded correctly or contains expected content."
    )
    inputs = {
        "info_type": {
            "description": "Type of page info to get (e.g., 'title', 'url', 'content')",
            "type": "string",
            "nullable": True,
        }
    }
    output_type = "string"
    
    def __init__(self, env: Any):
        self.env = env
        super().__init__()
    
    def forward(self, info_type: Optional[str] = None) -> str:
        return self.__call__(info_type)
    
    def __call__(self, info_type: Optional[str] = None) -> str:
        try:
            config = {}
            if info_type:
                config["info_type"] = info_type
                
            result = get_page_info(self.env, config)
            
            if result is None:
                return "Error: Failed to get page info"
            
            return str(result)
        except Exception as e:
            return f"Error getting page info: {str(e)}"
    
    def to_code_prompt(self) -> str:
        return (
            "def get_page_info(info_type: str = None) -> str:\n"
            "    '''Get detailed information about the current page in Chrome.'''\n"
        )
