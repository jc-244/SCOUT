import json
from typing import Any, Dict, Optional
from smolagents import Tool


class GetVSCodeWorkspaceSettingValueByIntentTool(Tool):
    name = "get_vscode_workspace_setting_value_by_intent"
    description = (
        "Read a cached VS Code workspace .code-workspace file on the host and return values by intent (keyless). "
        "Looks under the 'settings' object inside the workspace JSON (data.settings in some contexts)."
    )
    inputs = {
        "config_file_path": {
            "type": "string",
            "description": "Host path to the cached workspace file (from get_vscode_workspace_file).",
        },
        "intent": {
            "type": "string",
            "description": (
                "One of: 'word_wrap_column', 'auto_save', 'auto_save_delay', 'exclude_pycache_exists'"
            ),
        },
    }
    output_type = "object"

    def __init__(self):
        super().__init__()
        self.env = None

    def _load_json(self, path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
        try:
            with open(path, "r", encoding="utf-8") as f:
                _ = f.readline()
                return json.load(f)
        except Exception:
            return None

    def _get_settings_obj(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        settings = data.get("settings")
        return settings if isinstance(settings, dict) else None

    def _extract(self, settings: Dict[str, Any], intent: str) -> Dict[str, Any]:
        try:
            if intent == "auto_save":
                value = settings.get("files.autoSave")
                return {"intent": intent, "value": value, "ok": value is not None}
            elif intent == "auto_save_delay":
                value = settings.get("files.autoSaveDelay")
                return {"intent": intent, "value": value, "ok": value is not None}
            elif intent == "word_wrap_column":
                value = settings.get("editor.wordWrapColumn")
                return {"intent": intent, "value": value, "ok": value is not None}
            elif intent == "exclude_pycache_exists":
                excludes = settings.get("files.exclude")
                exists = False
                if isinstance(excludes, dict):
                    exists = (excludes.get("**/__pycache__") is True)
                details = "files.exclude contains '**/__pycache__': true" if exists else "Not found"
                return {"intent": intent, "value": exists, "ok": exists, "details": details}
            else:
                return {"intent": intent, "ok": False, "error": f"Unsupported intent: {intent}"}
        except Exception as e:
            return {"intent": intent, "ok": False, "error": f"Extraction error: {str(e)}"}

    def forward(self, config_file_path: str, intent: str) -> Dict[str, Any]:
        try:
            data = self._load_json(config_file_path)
            if data is None:
                return {"intent": intent, "ok": False, "error": "Failed to read JSON or file not found"}
            settings = self._get_settings_obj(data)
            if settings is None:
                return {"intent": intent, "ok": False, "error": "Workspace file missing 'settings' object"}
            return self._extract(settings, intent)
        except Exception as e:
            return {"intent": intent, "ok": False, "error": f"Failed to parse workspace: {str(e)}"}

    def __call__(self, config_file_path: str, intent: str) -> Dict[str, Any]:
        return self.forward(config_file_path=config_file_path, intent=intent)

    def to_code_prompt(self) -> str:
        return (
            f"Use {self.name} to read workspace settings by intent from a cached .code-workspace file.\n"
            "First call get_vscode_workspace_file to cache the file, then:\n"
            "- get_vscode_workspace_setting_value_by_intent(config_file_path=\"/path/project.code-workspace\", intent=\"word_wrap_column\")\n"
            "- get_vscode_workspace_setting_value_by_intent(config_file_path=\"/path/project.code-workspace\", intent=\"exclude_pycache_exists\")\n"
        )
