from smolagents import Tool
from desktop_env.evaluators.getters.vlc import get_default_video_player


class GetDefaultVideoPlayerTool(Tool):
    name = "get_default_video_player"
    description = """Get the default video player application for the system.
    """
    inputs = {}
    output_type = "string"

    def __init__(self, env):
        super().__init__()
        self.env = env

    def forward(self) -> str:
        try:
            config = {}
            result = get_default_video_player(self.env, config)
            return result if result else "unknown"
        except Exception as e:
            return f"Error: Failed to get default video player: {str(e)}"

    def __call__(self) -> str:
        return self.forward()

    def to_code_prompt(self) -> str:
        return f"""You can use the {self.name} tool to get the system's default video player.
Example usage:
- get_default_video_player()

The tool returns the name of the default video player application (e.g., 'vlc.desktop', 'totem.desktop').
Note: Currently only supports Linux systems."""
