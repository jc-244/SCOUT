from smolagents import Tool
from typing import Any, Dict, Optional
import os


class GetHostFileContentTool(Tool):
    
    name = "get_host_file_content"
    description = (
        "Get the content of a file on the host. Returns the file content as text."
    )
    inputs = {
        "file_path": {
            "description": "Absolute path to the file on the host",
            "type": "string",
        },
        "encoding": {
            "description": "Text encoding to use (default: 'utf-8')",
            "type": "string",
            "nullable": True,
        }
    }
    output_type = "string"
    
    def __init__(self):
        super().__init__()
    
    def forward(self, file_path: str, encoding: Optional[str] = None) -> str:
        return self.__call__(file_path, encoding)
    
    def __call__(self, file_path: str, encoding: Optional[str] = None) -> str:
        try:
            if not os.path.exists(file_path):
                return f"Error: File not found: '{file_path}'"
            
            if not os.path.isfile(file_path):
                return f"Error: Path is not a file: '{file_path}'"
            
            with open(file_path, 'r', encoding=encoding or 'utf-8') as f:
                content = f.read()
            print(content)
            return content
        except PermissionError:
            return f"Error: Permission denied when accessing file: '{file_path}'"
        except UnicodeDecodeError:
            return f"Error: Could not decode file with the specified encoding: '{file_path}'"
        except Exception as e:
            return f"Error getting content from host file: '{file_path}': {str(e)}"
    
    def to_code_prompt(self) -> str:
        return (
            "def get_host_file_content(file_path: str, encoding: str = None) -> str:\n"
            "    '''Get the content of a file on the host. and it will print it first and then return the content string''',\n"
        )