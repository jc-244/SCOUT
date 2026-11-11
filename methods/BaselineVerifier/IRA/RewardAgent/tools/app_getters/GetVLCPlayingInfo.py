from smolagents import Tool
from desktop_env.evaluators.getters.vlc import get_vlc_playing_info


class GetVLCPlayingInfoTool(Tool):
    name = "get_vlc_playing_info"
    description = """Get current playing information from VLC's HTTP interface.
    
    This tool retrieves status information about what VLC is currently playing, including playback status, current media info, and control information.
    Requires VLC HTTP interface to be enabled.
    """
    inputs = {
        "dest": {
            "type": "string", 
            "description": "Destination filename to save the VLC status XML"
        }
    }
    output_type = "string"

    def __init__(self, env):
        super().__init__()
        self.env = env

    def forward(self, dest: str) -> str:
        try:
            config = {"dest": dest}
            result = get_vlc_playing_info(self.env, config)
            if result is None:
                return "Error: Failed to get VLC playing info - check if VLC HTTP interface is enabled"
            return f"VLC playing info saved to {result} on the host"
        except Exception as e:
            return f"Error: Failed to get VLC playing info: {str(e)}"

    def __call__(self, dest: str) -> str:
        return self.forward(dest)

    def to_code_prompt(self) -> str:
        return f"""You can use the {self.name} tool to get VLC's current playing status.
Example usage:
- get_vlc_playing_info(dest="vlc_status.xml")
- get_vlc_playing_info(dest="current_playback.xml")

The tool returns the path to the saved XML file containing VLC status information.
Note: VLC HTTP interface must be enabled for this to work."""
