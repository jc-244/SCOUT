from typing import List, Dict, Any, Optional


BROWSE_TOOL = {
    "type": "function",
    "function": {
        "name": "browse_url",
        "description": "Opens a new browser window and navigates to the specified URL. Use this for opening web pages.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to open, e.g., 'https://www.google.com'."
                }
            },
            "required": ["url"]
        }
    }
}

EXECUTE_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": "Executes a snippet of Python code in the environment. Use it for calculations, string manipulation, or logic that doesn't require direct GUI interaction. The code is sandboxed and cannot interact with the GUI directly.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute. For example, 'print(1+1)'."
                }
            },
            "required": ["code"]
        }
    }
}

EXECUTE_SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_shell",
        "description": "Executes a shell (bash) command in the environment's terminal. Use it for file system operations (ls, cd, mkdir), checking software versions, or running command-line tools.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute. For example, 'ls -l'."
                }
            },
            "required": ["command"]
        }
    }
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search for information on the internet",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keywords"}
            },
            "required": ["query"]
        }
    }
}

VISIT_TOOL = {
    "type": "function",
    "function": {
        "name": "visit",
        "description": "Retrieves the full text content of web pages or PDFs from a list of URLs.",
        "parameters": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "An array of URLs to visit and extract content from."
                }
            },
            "required": ["urls"]
        }
    }
}

CHECK_SCREENSHOT_TOOL = {
    "type": "function",
    "function": {
        "name": "check_screenshot",
        "description": "View specific screenshot of one step from the trajectory to check if certain conditions are met. Use this tool to examine screenshot at particular step (e.g., step_1, step_7, step_10) to verify task progress or completion.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of screenshot names to view, formatted as 'step_N' where N is the step number (e.g., ['step_7']). Only one step is allowed.",
                    "minItems": 1,
                    "maxItems": 1
                }
            },
            "required": ["image_names"]
        }
    }
}


TOOL_REGISTRY = {
    "browse": BROWSE_TOOL,
    "execute_python": EXECUTE_PYTHON_TOOL,
    "execute_shell": EXECUTE_SHELL_TOOL,
    "search": SEARCH_TOOL,
    "visit": VISIT_TOOL,
    "check_screenshot": CHECK_SCREENSHOT_TOOL,
}

def get_tool_definitions(
    tool_names: List[str],
) -> List[Dict[str, Any]]:
    definitions = []
    for name in tool_names:
        if name in TOOL_REGISTRY:
            definitions.append(TOOL_REGISTRY[name])
        else:
            print(f"Warning: Tool '{name}' not found in registry and is not a dynamic tool. Skipping.")
    return definitions