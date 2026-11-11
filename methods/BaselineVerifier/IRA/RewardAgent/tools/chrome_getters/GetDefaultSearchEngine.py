from smolagents import Tool
from typing import Any, Dict
import os


from desktop_env.evaluators.getters.chrome import get_default_search_engine


class GetDefaultSearchEngineTool(Tool):
    
    name = "get_default_search_engine"
    description = (
        "Get the default search engine currently set in Chrome. "
        "This can help verify if the user has changed the search engine settings. "
        "Returns the name/URL of the default search engine."
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
            result = get_default_search_engine(self.env, config)
            
            if result is None:
                return "Error: Failed to get default search engine"
            
            return str(result)
        except Exception as e:
            return f"Error getting default search engine: {str(e)}"
    
    def to_code_prompt(self) -> str:
        return (
            "def get_default_search_engine() -> str:\n"
            "    '''Get Chrome's current default search engine.'''\n"
        )
