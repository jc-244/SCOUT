from smolagents import Tool
from typing import Any, Dict
import os


from desktop_env.evaluators.getters.chrome import get_open_tabs_info


class GetOpenTabsInfoTool(Tool):
    
    name = "get_open_tabs_info"
    description = (
        "Get information about all currently open tabs in Chrome. "
        "This can help verify if tabs were opened/closed correctly or check tab content. "
        "Returns tab information including URLs and titles."
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
            result = get_open_tabs_info(self.env, config)
            
            if result is None:
                return "Error: Failed to get open tabs info"
            
            return str(result)
        except Exception as e:
            return f"Error getting open tabs info: {str(e)}"
    
    def to_code_prompt(self) -> str:
        return (
            "def get_open_tabs_info() -> str:\n"
            "    '''Get information about currently open tabs in Chrome.'''\n"
        )
