from smolagents import Tool


class GetVSCodeUserSettingsFileTool(Tool):
    name = "get_vscode_user_settings_file"
    description = """
    "Fetch the VS Code *User-level* settings.json from the Virtual Machine (VM) and cache it on the host. "
    "IMPORTANT: This file ONLY records settings explicitly modified by the user. It does NOT represent "
    "the complete effective VS Code configuration. Any setting not present in this file should be "
    "assumed to retain its default value provided by VS Code or extensions. "
    "The absence of a setting key does NOT indicate that the feature is disabled or removed. "
    "This tool strictly copies the file and does NOT resolve defaults, extension settings, or platform-specific behavior."
    
    """
    inputs = {
        "dest": {
            "type": "string",
            "description": "Destination filename to save in host cache.",
        },
    }
    output_type = "string"

    def __init__(self, env):
        super().__init__()
        self.env = env

    def forward(self , dest: str) -> str:
        vm_user = "user" 
        
        try:
            code_dir =  "Code"
            vm_path = f"/home/{vm_user}/.config/{code_dir}/User/settings.json"
            file_bytes = self.env.controller.get_file(vm_path)
            if file_bytes is None:
                return "Error: Failed to fetch settings.json from VM (file not found or copy failed)"
            import os
            os.makedirs(self.env.cache_dir, exist_ok=True)
            host_path = os.path.join(self.env.cache_dir, dest)
            with open(host_path, "wb") as f:
                f.write(file_bytes)
            return host_path
        except Exception as e:
            return f"Error: Failed to get VS Code settings file: {str(e)}"

    def __call__(self, dest) -> str:
        return self.forward(dest=dest)

    def to_code_prompt(self) -> str:
        return (
            f"You can use {self.name} to copy VS Code User settings.json from the VM to the host cache.\n"
            "Examples:\n"
            "- get_vscode_user_settings_file( dest=\"settings.json\")\n"
        )
