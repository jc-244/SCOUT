from smolagents import Tool
from .utils import load_prefs_map, get_pref_value, to_json_string


class GetThunderbirdSmtpServerInfoTool(Tool):
    name = "get_thunderbird_smtp_server_info"
    description = (
        "Return Thunderbird SMTP server info as JSON from prefs.js for a given smtp key (e.g., 'smtp1'). "
        "Use get_thunderbird_prefs_file first if the file is inside the VM."
    )
    inputs = {
        "config_file_path": {
            "type": "string",
            "description": "Host path to Thunderbird prefs.js",
        },
        "smtp_key": {
            "type": "string",
            "description": "SMTP key like 'smtp1' (defaults to 'smtp1')",
        },
    }
    output_type = "string"

    def forward(self, config_file_path: str, smtp_key: str = "smtp1") -> str:
        try:
            prefs = load_prefs_map(config_file_path)
            base = f"mail.smtpserver.{smtp_key}"

            hostname = get_pref_value(prefs, f"{base}.hostname")
            port = get_pref_value(prefs, f"{base}.port")
            try_ssl = get_pref_value(prefs, f"{base}.try_ssl")
            username = get_pref_value(prefs, f"{base}.username")
            description = get_pref_value(prefs, f"{base}.description")
            authMethod = get_pref_value(prefs, f"{base}.authMethod")
            clientid = get_pref_value(prefs, f"{base}.clientid")

            result = {
                "hostname": hostname if hostname is not None else None,
                "port": port if port is not None else None,
                "try_ssl": try_ssl if try_ssl is not None else None,
                "username": username if username is not None else None,
                "description": description if description is not None else None,
                "authMethod": authMethod if authMethod is not None else None,
                "clientid": clientid if clientid is not None else None,
            }

            return to_json_string(result)
        except Exception as e:
            return f"Error: {e}"

    def __call__(self, config_file_path: str, smtp_key: str = "smtp1") -> str:
        return self.forward(config_file_path, smtp_key)

    def to_code_prompt(self) -> str:
        return (
            "def get_thunderbird_smtp_server_info(config_file_path: str, smtp_key: str = 'smtp1') -> str:\n"
            "    '''Return SMTP server info JSON with fixed fields for the given smtp key.'''\n"
        )
