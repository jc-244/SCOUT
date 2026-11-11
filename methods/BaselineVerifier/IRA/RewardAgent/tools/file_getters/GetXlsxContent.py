from smolagents import Tool
from typing import Any, Dict, Optional
import os


from desktop_env.evaluators.getters.file import get_content_from_vm_file


class GetXlsxContentTool(Tool):
    
    name = "get_xlsx_content"
    description = (
        "Get the content of a file from the virtual machine (VM). Returns the file content as text."
    )
    inputs = {
        "vm_path": {
            "description": "Absolute path to the file on the virtual machine (VM)",
            "type": "string",
        },
        "encoding": {
            "description": "Text encoding to use (default: 'utf-8')",
            "type": "string",
            "nullable": True,
        }
    }
    output_type = "string"
    
    def __init__(self, env: Any):
        self.env = env
        super().__init__()
    
    def forward(self, vm_path: str, encoding: Optional[str] = None) -> str:
        return self.__call__(vm_path, encoding)
    
    def __call__(self, vm_path: str, encoding: Optional[str] = None) -> str:
        try:
            config = {"path": vm_path}
            if encoding:
                config["encoding"] = encoding
                
            result = get_content_from_vm_file(self.env, config)
            
            if result is None:
                return f"Error: Failed to get content from file on the virtual machine (VM): '{vm_path}'. Please ensure you're using a valid path that exists on the VM, not a host path."
            
            return str(result)
        except Exception as e:
            return f"Error getting content from file on the virtual machine (VM): '{vm_path}': {str(e)}. Please ensure you're using a valid path that exists on the VM, not a host path."
    
    def to_code_prompt(self) -> str:
        return (
            "def get_xlsx_content(vm_path: str, encoding: str = None) -> str:\n"
            "    '''Get the content of a file from the virtual machine (VM).'''\n"
        )
