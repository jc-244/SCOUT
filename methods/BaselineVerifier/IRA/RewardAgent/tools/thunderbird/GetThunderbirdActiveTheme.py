from smolagents import Tool
from .utils import load_prefs_map, get_pref_value


class GetThunderbirdActiveThemeTool(Tool):
    name = "get_thunderbird_active_theme"
    description = (
        "Return Thunderbird active theme ID from prefs.js (extensions.activeThemeID). "
        "Use get_thunderbird_prefs_file first if the file is inside the VM."
    )
    inputs = {
        "config_file_path": {
            "type": "string",
            "description": "Host path to Thunderbird prefs.js",
        }
    }
    output_type = "string"

    def forward(self, config_file_path: str) -> str:
        try:
            prefs = load_prefs_map(config_file_path)
            val = get_pref_value(prefs, "extensions.activeThemeID")
            if val is None:
                return "Error: extensions.activeThemeID not found"
            return str(val)
        except Exception as e:
            return f"Error: {e}"

    def __call__(self, config_file_path: str) -> str:
        return self.forward(config_file_path)

    def to_code_prompt(self) -> str:
        return (
            "def get_thunderbird_active_theme(config_file_path: str) -> str:\n"
            "    '''Return the active theme ID from prefs.js (extensions.activeThemeID).'''\n"
        )
