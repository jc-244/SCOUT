from smolagents import Tool
from typing import Any, Dict, Optional
import os


from desktop_env.evaluators.getters.chrome import get_bookmarks


class GetBookmarksTool(Tool):
    
    name = "get_bookmarks"
    description = (
        "Get bookmarks data from Chrome. This can help verify if bookmarks were "
        "added, removed, or organized correctly. Optionally filter by bookmark name. "
        "Returns bookmark information as a string."
    )
    inputs = {
        "bookmark_name": {
            "description": "Optional: specific bookmark name to search for",
            "type": "string",
            "nullable": True,
        }
    }
    output_type = "string"
    
    def __init__(self, env: Any):
        self.env = env
        super().__init__()
    
    def forward(self, bookmark_name: Optional[str] = None) -> str:
        return self.__call__(bookmark_name)
    
    def __call__(self, bookmark_name: Optional[str] = None) -> str:
        try:
            config = {}
            if bookmark_name:
                config["bookmark_name"] = bookmark_name
                
            result = get_bookmarks(self.env, config)
            
            if result is None:
                return "Error: Failed to get bookmarks data"
            
            return str(result)
        except Exception as e:
            return f"Error getting bookmarks data: {str(e)}"
    
    def to_code_prompt(self) -> str:
        return (
            "def get_bookmarks(bookmark_name: str = None) -> str:\n"
            "    '''Get bookmarks data from Chrome, optionally filtered by name.'''\n"
        )
