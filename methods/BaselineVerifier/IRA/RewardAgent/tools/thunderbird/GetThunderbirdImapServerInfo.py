from smolagents import Tool
from .utils import load_prefs_map, get_pref_value, to_json_string


class GetThunderbirdImapServerInfoTool(Tool):
    name = "get_thunderbird_imap_server_info"
    description = (
        "Return Thunderbird IMAP server info as JSON from prefs.js for a given server key (e.g., 'server1'). "
        "Use get_thunderbird_prefs_file first if the file is inside the VM."
    )
    inputs = {
        "config_file_path": {
            "type": "string",
            "description": "Host path to Thunderbird prefs.js",
        },
        "server_key": {
            "type": "string",
            "description": "Server key like 'server1' (defaults to 'server1')",
        },
    }
    output_type = "string"

    def forward(self, config_file_path: str, server_key: str = "server1") -> str:
        try:
            prefs = load_prefs_map(config_file_path)
            base = f"mail.server.{server_key}"

            hostname = get_pref_value(prefs, f"{base}.hostname")
            port = get_pref_value(prefs, f"{base}.port")
            socketType = get_pref_value(prefs, f"{base}.socketType")
            typ = get_pref_value(prefs, f"{base}.type")
            userName = get_pref_value(prefs, f"{base}.userName")
            check_new_mail = get_pref_value(prefs, f"{base}.check_new_mail")
            max_cached_connections = get_pref_value(prefs, f"{base}.max_cached_connections")
            timeout = get_pref_value(prefs, f"{base}.timeout")
            trash_folder_name = get_pref_value(prefs, f"{base}.trash_folder_name")
            directory = get_pref_value(prefs, f"{base}.directory")
            namespace_personal = get_pref_value(prefs, f"{base}.namespace.personal")

            result = {
                "hostname": hostname if hostname is not None else None,
                "port": port if port is not None else None,
                "socketType": socketType if socketType is not None else None,
                "type": typ if typ is not None else None,
                "userName": userName if userName is not None else None,
                "check_new_mail": bool(check_new_mail) if isinstance(check_new_mail, bool) else (True if check_new_mail == "true" else (False if check_new_mail == "false" else None)),
                "max_cached_connections": max_cached_connections if max_cached_connections is not None else None,
                "timeout": timeout if timeout is not None else None,
                "trash_folder_name": trash_folder_name if trash_folder_name is not None else None,
                "directory": directory if directory is not None else None,
                "namespace_personal": namespace_personal if namespace_personal is not None else None,
            }

            return to_json_string(result)
        except Exception as e:
            return f"Error: {e}"

    def __call__(self, config_file_path: str, server_key: str = "server1") -> str:
        return self.forward(config_file_path, server_key)

    def to_code_prompt(self) -> str:
        return (
            "def get_thunderbird_imap_server_info(config_file_path: str, server_key: str = 'server1') -> str:\n"
            "    '''Return IMAP server info JSON with fixed fields for the given server key.'''\n"
        )
