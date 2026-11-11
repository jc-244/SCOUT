from smolagents import Tool
from .utils import load_prefs_map, get_pref_value


class GetThunderbirdUseSystemTimezoneTool(Tool):
    name = "get_thunderbird_use_system_timezone"
    description = (
        "Return whether Thunderbird uses system timezone from prefs.js (calendar.timezone.useSystemTimezone). "
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
            val = get_pref_value(prefs, "calendar.timezone.useSystemTimezone")
            if val is None:
                return "Error: calendar.timezone.useSystemTimezone not found"
            return "true" if bool(val) else "false"
        except Exception as e:
            return f"Error: {e}"

    def __call__(self, config_file_path: str) -> str:
        return self.forward(config_file_path)

    def to_code_prompt(self) -> str:
        return (
            "def get_thunderbird_use_system_timezone(config_file_path: str) -> str:\n"
            "    '''Return 'true' or 'false' from prefs.js (calendar.timezone.useSystemTimezone).'''\n"
        )
