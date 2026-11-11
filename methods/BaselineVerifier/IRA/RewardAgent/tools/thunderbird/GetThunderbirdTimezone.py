from smolagents import Tool
from .utils import load_prefs_map, get_pref_value


class GetThunderbirdTimezoneTool(Tool):
    name = "get_thunderbird_timezone"
    description = (
        "Return Thunderbird timezone from prefs.js (calendar.timezone.local). "
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
            val = get_pref_value(prefs, "calendar.timezone.local")
            if val is None:
                return "Error: calendar.timezone.local not found"
            return str(val)
        except Exception as e:
            return f"Error: {e}"

    def __call__(self, config_file_path: str) -> str:
        return self.forward(config_file_path)

    def to_code_prompt(self) -> str:
        return (
            "def get_thunderbird_timezone(config_file_path: str) -> str:\n"
            "    '''Return the timezone from prefs.js (calendar.timezone.local).'''\n"
        )
