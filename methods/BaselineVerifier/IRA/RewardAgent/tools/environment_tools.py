from smolagents import Tool
from typing import Any, Optional
import os


from desktop_env.evaluators.getters.general import (
    get_vm_command_line,
    get_vm_command_error,
    get_vm_terminal_output
)
from desktop_env.evaluators.getters.file import (
    get_vm_file,
    get_cache_file
)
from desktop_env.evaluators.getters.chrome import (
    get_active_url_from_accessTree
)


class VMCommandLineTool(Tool):
    
    name = "execute_vm_command"
    description = (
        "Execute read-only commands on the VM to OBSERVE and COLLECT system information. "
    "This tool should be used as an information-gathering interface, NOT as a way to "
    "assert or assume conclusions.\n\n"
    "Guidelines:\n"
    "- Use this tool to list, display, or inspect system state (e.g., list files, "
    "show configurations, print full command outputs).\n"
    "- Prefer commands that return COMPLETE and UNFILTERED information, rather than "
    "commands that rely on assumptions or hard-coded matches.\n"
    "- Avoid commands that directly encode expected answers (e.g., overly specific "
    "grep patterns). If filtering is needed, first retrieve the full output, then "
    "reason about it.\n"
    "- Treat empty or missing output as incomplete evidence, not as proof of absence.\n\n"
    "Typical use cases:\n"
    "- List installed extensions, packages, or processes\n"
    "- Read configuration files or system status outputs\n"
    "- Inspect directories or environment state\n\n"
    "Examples (preferred):\n"
    "- 'code --list-extensions'\n check vscode extenstions"
    "- 'ls /path/to/dir'\n check what's under the target dir"
    "-d /path/to/folder\n check if directory exists"
    "- 'cat /path/to/file'\n\n"
    "Examples (discouraged):\n"
    "- 'code --list-extensions | grep python'\n"
    "- 'ps aux | grep exact_process_name'\n\n"
    "The model should perform interpretation and verification based on the returned "
    "information, rather than embedding assumptions into the command itself."
    )
    inputs = {
        "command": {
            "description": "The shell command to execute (read-only operations only)",
            "type": "string",
        },
        "shell": {
            "description": "Whether to execute through shell (default: False)",
            "type": "boolean",
            "nullable": True,
        }
    }
    output_type = "string"
    
    def __init__(self, env: Any):
        self.env = env
        super().__init__()
    
    def forward(self, command: str, shell: bool = False) -> str:
        return self.__call__(command, shell)
    
    def __call__(self, command: str, shell: bool = False) -> str:
        try:
            config = {
                "command": command,
                "shell": shell
            }
            result = get_vm_command_line(self.env, config)
            
            if result is None:
                return f"Error: Failed to execute command '{command}'"
            
            return str(result)
        except Exception as e:
            return f"Error executing command '{command}': {str(e)}"
    
    def to_code_prompt(self) -> str:
        return (
            "def execute_vm_command(command: str, shell: bool = False) -> str:\n"
            "    '''Execute a read-only command on the VM to check system state. "
            "Returns command output.'''\n"
        )


class VMCommandErrorTool(Tool):
    
    name = "get_vm_command_error"
    description = (
        "Get the error output (stderr) from executing a command on the VM. "
        "Use this to check if a command produced any errors or warnings. "
        "Returns the error output as a string."
    )
    inputs = {
        "command": {
            "description": "The shell command to execute",
            "type": "string",
        },
        "shell": {
            "description": "Whether to execute through shell (default: False)",
            "type": "boolean",
            "nullable": True,
        }
    }
    output_type = "string"
    
    def __init__(self, env: Any):
        self.env = env
        super().__init__()
    
    def forward(self, command: str, shell: bool = False) -> str:
        return self.__call__(command, shell)
    
    def __call__(self, command: str, shell: bool = False) -> str:
        try:
            config = {
                "command": command,
                "shell": shell
            }
            result = get_vm_command_error(self.env, config)
            
            if result is None:
                return f"Error: Failed to get error output for command '{command}'"
            
            return str(result) if result else "(no error output)"
        except Exception as e:
            return f"Error getting command error for '{command}': {str(e)}"
    
    def to_code_prompt(self) -> str:
        return (
            "def get_vm_command_error(command: str, shell: bool = False) -> str:\n"
            "    '''Get error output (stderr) from VM command execution.'''\n"
        )


class VMFileTool(Tool):
    
    name = "get_vm_file"
    description = (
        "Retrieve a file from the virtual machine to inspect its contents. "
        "The file will be downloaded from the virtual machineto the host's cache directory and its path returned. "
        "Use this to verify file contents, check if files were created/modified correctly, etc. "
        "Returns the host path to the downloaded file."
    )
    inputs = {
        "vm_path": {
            "description": "Absolute path to the file on the virtual machine (e.g., '/home/user/Desktop/file.txt')",
            "type": "string",
        },
        "dest_name": {
            "description": "Destination filename in the host's cache (e.g., 'file.txt')",
            "type": "string",
        }
    }
    output_type = "string"
    
    def __init__(self, env: Any):
        self.env = env
        super().__init__()
    
    def forward(self, vm_path: str, dest_name: str) -> str:
        return self.__call__(vm_path, dest_name)
    
    def __call__(self, vm_path: str, dest_name: str) -> str:
        try:
            config = {
                "path": vm_path,
                "dest": dest_name
            }
            result = get_vm_file(self.env, config)
            
            if result is None:
                return f"Error: Failed to retrieve file '{vm_path}' from VM"
            
            return str(result)
        except Exception as e:
            return f"Error retrieving file '{vm_path}': {str(e)}"
    
    def to_code_prompt(self) -> str:
        return (
            "def get_vm_file(vm_path: str, dest_name: str) -> str:\n"
            "    '''Retrieve a file from the virtual machine. Returns host path to downloaded file.'''\n"
        )


class VMTerminalOutputTool(Tool):
    
    name = "get_terminal_output"
    description = (
        "Get the current terminal output/history from the VM. "
        "Use this to check what commands were executed and their results. "
        "Returns the terminal output as a string."
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
            result = get_vm_terminal_output(self.env, config)
            
            if result is None:
                return "Error: Failed to get terminal output"
            
            return str(result)
        except Exception as e:
            return f"Error getting terminal output: {str(e)}"
    
    def to_code_prompt(self) -> str:
        return (
            "def get_terminal_output() -> str:\n"
            "    '''Get the current terminal output/history from the VM.'''\n"
        )


class GetActiveURLTool(Tool):
    name = "get_active_url"
    description = "Get the URL of the currently active Chrome tab."
    inputs = {
        "goto_prefix": {
            "description": "URL prefix to add to the beginning (default: 'https://').",
            "type": "string",
            "nullable": True,
        }
    }
    output_type = "string"
    
    def __init__(self, env):
        self.env = env
        super().__init__()

    def forward(self, goto_prefix: Optional[str] = None):
        return self.__call__(goto_prefix)

    def __call__(self, goto_prefix: Optional[str] = None):
        try:
            if goto_prefix is None:
                goto_prefix = "https://"

            config = {"goto_prefix": goto_prefix}
            result = get_active_url_from_accessTree(self.env, config)
            
            if result is None:
                return "Error: Failed to get active URL from accessibility tree."
            
            return str(result)
        except Exception as e:
            return f"Error getting active URL: {str(e)}"
    
    def to_code_prompt(self) -> str:
        return (
            "def get_active_url(goto_prefix: str = 'https://') -> str:\n"
            "    '''Get the URL of the currently active Chrome tab.'''\n"
        )
