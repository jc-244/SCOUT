from smolagents import Tool
from .utils import load_prefs_map, get_pref_value, to_json_string


class GetThunderbirdAccountsTool(Tool):
    name = "get_thunderbird_accounts"
    description = (
        "Return Thunderbird accounts summary as JSON from prefs.js. "
        "Includes accounts list, defaultAccount, localFoldersServer, and account->server mapping. "
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
            accounts_raw = get_pref_value(prefs, "mail.accountmanager.accounts")
            accounts_list = []
            if isinstance(accounts_raw, str) and accounts_raw:
                accounts_list = [a.strip() for a in accounts_raw.split(",") if a.strip()]

            default_account = get_pref_value(prefs, "mail.accountmanager.defaultaccount")
            if default_account is None:
                default_account = None

            local_folders_server = get_pref_value(prefs, "mail.accountmanager.localfoldersserver")
            if local_folders_server is None:
                local_folders_server = None

            mapping = {}
            for acc in accounts_list:
                server_key = get_pref_value(prefs, f"mail.account.{acc}.server")
                mapping[acc] = server_key if server_key is not None else None

            result = {
                "accounts": accounts_list,
                "defaultAccount": default_account,
                "localFoldersServer": local_folders_server,
                "accountServerMap": mapping,
            }
            return to_json_string(result)
        except Exception as e:
            return f"Error: {e}"

    def __call__(self, config_file_path: str) -> str:
        return self.forward(config_file_path)

    def to_code_prompt(self) -> str:
        return (
            "def get_thunderbird_accounts(config_file_path: str) -> str:\n"
            "    '''Return accounts summary JSON: accounts, defaultAccount, localFoldersServer, accountServerMap.'''\n"
        )
