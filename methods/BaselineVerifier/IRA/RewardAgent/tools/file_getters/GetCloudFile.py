from smolagents import Tool
from typing import Any, Dict, Optional
import os


from desktop_env.evaluators.getters.file import get_cloud_file


class GetCloudFileTool(Tool):
    
    name = "get_cloud_file"
    description = (
        "Download and retrieve a file from cloud storage (e.g., Google Drive, etc.). "
        "This can help access reference files or compare with downloaded content. "
        "Returns the local path to the downloaded file."
    )
    inputs = {
        "url": {
            "description": "URL or path to the cloud file",
            "type": "string",
        },
        "dest_name": {
            "description": "Destination filename for the downloaded file",
            "type": "string",
        }
    }
    output_type = "string"
    
    def __init__(self, env: Any):
        self.env = env
        super().__init__()
    
    def forward(self, url: str, dest_name: str) -> str:
        return self.__call__(url, dest_name)
    
    def __call__(self, url: str, dest_name: str) -> str:
        try:
            config = {
                "path": url,
                "dest": dest_name
            }
            result = get_cloud_file(self.env, config)
            
            if result is None:
                return f"Error: Failed to download cloud file from '{url}'"
            
            return str(result)
        except Exception as e:
            return f"Error downloading cloud file from '{url}': {str(e)}"
    
    def to_code_prompt(self) -> str:
        return (
            "def get_cloud_file(url: str, dest_name: str) -> str:\n"
            "    '''Download and retrieve a file from cloud storage.'''\n"
        )
