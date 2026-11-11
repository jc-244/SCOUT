from __future__ import annotations

from smolagents import Tool
from desktop_env.evaluators.getters.file import get_vm_file


class GetThunderbirdPrefsFileTool(Tool):
    name = "get_thunderbird_prefs_file"
    description = (
        "Download Thunderbird prefs.js from the VM to the Host cache directory. "
        "Use this before other thunderbird getters. Returns the Host path to the downloaded file. "
        "Defaults VM path to '/home/user/.thunderbird/t5q2a5hp.default-release/prefs.js'."
    )
    inputs = {
        "dest": {
            "type": "string",
            "description": (
                "Destination filename to save under the Host cache directory (env.cache_dir). "
                "Example: 'prefs.js'"
            ),
        }
    }
    output_type = "string"

    def __init__(self, env):
        super().__init__()
        self.env = env

    def forward(self, dest: str) -> str:
        try:
            vm_path = "/home/user/.thunderbird/t5q2a5hp.default-release/prefs.js"
            dest = dest or "prefs.js"
            host_path = get_vm_file(self.env, {"path": vm_path, "dest": dest})
            if not host_path:
                return "Error: Failed to download prefs.js from VM"
            return host_path
        except Exception as e:
            return f"Error: {e}"

    def __call__(self, dest: str) -> str:
        return self.forward(dest)

    def to_code_prompt(self) -> str:
        return (
            "def get_thunderbird_prefs_file(dest: str = 'prefs.js') -> str:\n"
            "    '''Download Thunderbird prefs.js from VM to Host cache and return the Host path.'''\n"
        )
