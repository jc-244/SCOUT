from smolagents import Tool
from desktop_env.evaluators.getters.gimp import get_gimp_config_file


class GetGimpConfigFileTool(Tool):
    name = "get_gimp_config_file"
    description = """Get GIMP configuration files to check settings.
    
    Useful for checking GIMP preferences, tool settings, and other configuration options.
    """
    inputs = {
        "file_name": {
            "type": "string", 
            "description": "Name of the GIMP config file to retrieve (e.g., 'gimprc', 'toolrc', 'sessionrc')"
        },
        "dest": {
            "type": "string", 
            "description": "Destination filename to save the config file"
        }
    }
    output_type = "string"

    def __init__(self, env):
        super().__init__()
        self.env = env

    def forward(self, file_name: str, dest: str) -> str:
        try:
            config = {"file_name": file_name, "dest": dest}
            result = get_gimp_config_file(self.env, config)
            if result is None:
                return "Error: Failed to get GIMP config file - file may not exist or GIMP not configured"
            return result
        except Exception as e:
            return f"Error: Failed to get GIMP config file: {str(e)}"

    def __call__(self, file_name: str, dest: str) -> str:
        return self.forward(file_name, dest)

    def to_code_prompt(self) -> str:
        return f"""You can use the {self.name} tool to get GIMP configuration files.
Example usage:
- get_gimp_config_file(file_name="gimprc", dest="gimp_preferences.txt")
- get_gimp_config_file(file_name="toolrc", dest="gimp_tools.txt")
- get_gimp_config_file(file_name="sessionrc", dest="gimp_session.txt")

The tool returns the path to the saved GIMP config file.
Note: Currently only supports Linux systems. Config files are in /home/user/.config/GIMP/2.10/"""
