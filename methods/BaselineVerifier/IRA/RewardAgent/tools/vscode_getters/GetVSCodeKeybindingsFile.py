from smolagents import Tool
import os


class GetVSCodeKeybindingsFileTool(Tool):
    name = "get_vscode_keybindings_file"
    description = (
        "Fetch the VS Code *User-level* keybindings.json from the VM and cache it on the host. "
    "IMPORTANT: This file ONLY contains user-defined keybinding overrides (diffs), NOT the full set "
    "of effective VS Code keybindings. Any keybinding that does NOT appear in this file should be "
    "assumed to still exist with its default behavior, unless explicitly unbound (e.g., with "
    "\"command\": \"-<commandId>\"). "
    "Absence of a keybinding entry (e.g., Ctrl+F) does NOT mean it was removed or disabled. "
    "The tool only copies the file and does NOT parse, merge, or resolve default keybindings."

    )
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

    def forward(self, dest: str) -> str:
        vm_user = "user"
        try:
            vm_path = f"/home/{vm_user}/.config/Code/User/keybindings.json"
            controller = getattr(self.env, "controller", None)
            if controller is None:
                return "Error: Missing env.controller"
            file_bytes = controller.get_file(vm_path)
            if file_bytes is None:
                return "Error: Failed to fetch keybindings.json from VM (file not found or copy failed)"
            os.makedirs(self.env.cache_dir, exist_ok=True)
            host_path = os.path.join(self.env.cache_dir, dest)
            with open(host_path, "wb") as f:
                f.write(file_bytes)
            return host_path
        except Exception as e:
            return f"Error: Failed to get VS Code keybindings file: {str(e)}"

    def __call__(self, dest: str) -> str:
        return self.forward(dest=dest)

    def to_code_prompt(self) -> str:
        return (
            f"Use {self.name} to copy VS Code User keybindings.json from the VM to the host cache.\n"
            "Example:\n"
            "- get_vscode_keybindings_file(dest=\"keybindings.json\")\n"
        )
