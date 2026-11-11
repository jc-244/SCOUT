from smolagents import Tool
from typing import Any, Dict
import os


from desktop_env.evaluators.getters.chrome import get_chrome_font_size


class GetChromeFontSizeTool(Tool):
    
    name = "get_chrome_font_size"
    description = (
        "Get the current font size setting in Chrome. "
        "This can help verify if font size was changed correctly. "
        "Returns the current font size value."
    )
    inputs = {}
    output_type = "string"
    
    def __init__(self, env: Any):
        self.env = env
        super().__init__()
    
    def forward(self) -> str:
        return self.__call__()
    
    def __call__(self) -> str:
        try:
            config = {}
            result = get_chrome_font_size(self.env, config)
            
            if result is None:
                return "Error: Failed to get Chrome font size"
            
            return str(result)
        except Exception as e:
            return f"Error getting Chrome font size: {str(e)}"
    
    def to_code_prompt(self) -> str:
        return (
            "def get_chrome_font_size() -> str:\n"
            "    '''Get Chrome's current font size setting.'''\n"
        )
