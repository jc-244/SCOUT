from smolagents import Tool
from desktop_env.evaluators.getters.vlc import get_vlc_config
import os


class GetVLCConfigTool(Tool):
    name = "get_vlc_config"
    description = """Get VLC configuration file to check settings.
    
    Useful for verifying VLC settings and preferences.
    """
    inputs = {
        "dest": {
            "type": "string", 
            "description": "Destination filename to save the VLC config file"
        }
    }
    output_type = "string"

    def __init__(self, env):
        super().__init__()
        self.env = env

    def forward(self, dest: str) -> str:
        try:
            config = {"dest": dest}
            _path = get_vlc_config(self.env, config)
            
            if os.path.exists(_path):
                with open(_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('input-record-path='):
                            return line
                        elif line.startswith('#input-record-path='):
                            return line
                return f"Error: input-record-path not found in VLC config file at {_path}"
            else:
                return f"Error: VLC config file not found at {_path}"
        except Exception as e:
            return f"Error: Failed to get VLC config: {str(e)}"

    def __call__(self, dest: str) -> str:
        return self.forward(dest)

    def to_code_prompt(self) -> str:
        return f"""You can use the {self.name} tool to get VLC's configuration file.
Example usage:
- get_vlc_config(dest="vlcrc.conf")
- get_vlc_config(dest="vlc_settings.txt")

The tool returns the input-record-path variable and its value from the VLC configuration file.
Config location varies by OS: /home/user/.config/vlc/vlcrc (Linux), /home/user/Library/Preferences/org.videolan.vlc/vlcrc (macOS), %APPDATA%\\vlc\\vlcrc (Windows)."""