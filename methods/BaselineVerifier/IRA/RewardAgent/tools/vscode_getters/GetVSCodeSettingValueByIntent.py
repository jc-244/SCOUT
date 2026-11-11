import json
from typing import Any, Dict, Optional
from smolagents import Tool


class GetVSCodeSettingValueByIntentTool(Tool):
    name = "get_vscode_setting_value_by_intent"
    description = """Read a cached VS Code User settings.json on the host and return values by intent (keyless or raw key/path).

    This tool parses the settings.json file that was previously copied from the VM to the host cache
    and returns the value for a given intent. In addition to predefined intents, it supports arbitrary
    VS Code settings keys and dotted paths. If a requested key/path cannot be resolved, the tool returns
    the available top-level keys to aid decision-making.

    Supported predefined intents (backward compatible):
    - auto_save: returns files.autoSave
    - auto_save_delay: returns files.autoSaveDelay
    - wrap_tabs: returns workbench.editor.wrapTabs
    - color_theme: returns workbench.colorTheme
    - python_missing_imports_severity: returns python.analysis.diagnosticSeverityOverrides.reportMissingImports
    - word_wrap_column: returns editor.wordWrapColumn
    - exclude_pycache_exists: checks files.exclude contains "**/__pycache__": true
    - editor_lineNumbers: returns editor.lineNumbers 

    Generic resolution (new):
    - If intent equals a top-level key (e.g., "workbench.colorTheme"), returns that value.
    - If intent is a dotted path (e.g., "python.analysis.diagnosticSeverityOverrides.reportMissingImports"):
      * Tries to match the longest dotted prefix as a top-level key, then descends into remaining path if the value is an object.
      * Also attempts direct nested traversal from root for settings stored as nested objects.
    - On failure, returns a list of available top-level keys (and nested keys if a prefix was matched) to help choose a valid key.
    """
    inputs = {
        "config_file_path": {
            "type": "string",
            "description": "Host path to the cached VS Code settings.json (from get_vscode_user_settings_file).",
        },
        "intent": {
            "type": "string",
            "description": (
                "Predefined intents (e.g., 'auto_save', 'color_theme') or a raw VS Code settings key/path. "
                "Examples: 'workbench.colorTheme', 'files.autoSave', "
                "'python.analysis.diagnosticSeverityOverrides.reportMissingImports'."
            ),
        },
    }
    output_type = "object"

    def __init__(self, env=None):
        super().__init__()
        self.env = env
        self._max_list_items = 200

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

    def _clip_list(self, items):
        if items is None:
            return None, 0, False
        try:
            items_list = sorted(list(items))
        except Exception:
            items_list = list(items)
        total = len(items_list)
        clipped = items_list[: self._max_list_items]
        return clipped, total, total > self._max_list_items

    def _extract_by_key_or_path(self, data: Dict[str, Any], key_or_path: str) -> Dict[str, Any]:
        try:
            if key_or_path in data:
                value = data.get(key_or_path)
                return {
                    "intent": key_or_path,
                    "value": value,
                    "ok": value is not None,
                    "details": "Resolved by exact top-level key",
                    "resolved_key": key_or_path,
                }
        except Exception:
            pass

        parts = key_or_path.split(".") if "." in key_or_path else [key_or_path]

        try:
            for i in range(len(parts), 0, -1):
                prefix = ".".join(parts[:i])
                if prefix in data:
                    prefix_val = data.get(prefix)
                    remaining = parts[i:]
                    if not remaining:
                        return {
                            "intent": key_or_path,
                            "value": prefix_val,
                            "ok": prefix_val is not None,
                            "details": "Resolved by dotted prefix top-level key",
                            "resolved_prefix": prefix,
                            "remaining_path": remaining,
                        }
                    if isinstance(prefix_val, dict):
                        current = prefix_val
                        for seg in remaining:
                            if isinstance(current, dict) and seg in current:
                                current = current.get(seg)
                            else:
                                nested_keys, nested_total, nested_truncated = self._clip_list(
                                    current.keys() if isinstance(current, dict) else []
                                )
                                available_keys, total, truncated = self._clip_list(data.keys())
                                details = "Unable to resolve remaining path from prefix"
                                if nested_truncated:
                                    details += f"; nested_keys truncated to {self._max_list_items} of {nested_total}"
                                if truncated:
                                    details += f"; available_keys truncated to {self._max_list_items} of {total}"
                                return {
                                    "intent": key_or_path,
                                    "ok": False,
                                    "error": f"Path segment not found: '{seg}'",
                                    "resolved_prefix": prefix,
                                    "remaining_path": remaining,
                                    "nested_keys": nested_keys,
                                    "available_keys": available_keys,
                                    "available_keys_count": total,
                                    "details": details,
                                }
                        return {
                            "intent": key_or_path,
                            "value": current,
                            "ok": current is not None,
                            "details": "Resolved by prefix + nested path",
                            "resolved_prefix": prefix,
                            "remaining_path": remaining,
                        }
                    available_keys, total, truncated = self._clip_list(data.keys())
                    details = "Prefix matched but value is not an object"
                    if truncated:
                        details += f"; available_keys truncated to {self._max_list_items} of {total}"
                    return {
                        "intent": key_or_path,
                        "ok": False,
                        "error": "Non-object prefix cannot resolve remaining path",
                        "resolved_prefix": prefix,
                        "remaining_path": remaining,
                        "available_keys": available_keys,
                        "available_keys_count": total,
                        "details": details,
                    }
        except Exception:
            pass

        try:
            current: Any = data
            traversed = []
            for seg in parts:
                if isinstance(current, dict) and seg in current:
                    current = current.get(seg)
                    traversed.append(seg)
                else:
                    break
            if len(traversed) == len(parts):
                return {
                    "intent": key_or_path,
                    "value": current,
                    "ok": current is not None,
                    "details": "Resolved by direct nested traversal",
                    "resolved_path": traversed,
                }
        except Exception:
            pass

        available_keys, total, truncated = self._clip_list(data.keys() if isinstance(data, dict) else [])
        details = "Key/path not found"
        if truncated:
            details += f"; available_keys truncated to {self._max_list_items} of {total}"
        return {
            "intent": key_or_path,
            "ok": False,
            "error": "Unsupported intent or key/path not present in settings.json",
            "available_keys": available_keys,
            "available_keys_count": total,
            "details": details,
        }

    def _extract(self, data: Dict[str, Any], intent: str) -> Dict[str, Any]:
        try:
            if intent == "auto_save":
                value = data.get("files.autoSave")
                return {"intent": intent, "value": value, "ok": value is not None}
            elif intent == "auto_save_delay":
                value = data.get("files.autoSaveDelay")
                return {"intent": intent, "value": value, "ok": value is not None}
            elif intent == "wrap_tabs":
                value = data.get("workbench.editor.wrapTabs")
                return {"intent": intent, "value": value, "ok": value is not None}
            elif intent == "color_theme":
                value = data.get("workbench.colorTheme")
                return {"intent": intent, "value": value, "ok": value is not None}
            elif intent == "python_missing_imports_severity":
                overrides = data.get("python.analysis.diagnosticSeverityOverrides", {})
                if isinstance(overrides, dict):
                    value = overrides.get("reportMissingImports")
                else:
                    value = None
                return {"intent": intent, "value": value, "ok": value is not None}
            elif intent == "word_wrap_column":
                value = data.get("editor.wordWrapColumn")
                return {"intent": intent, "value": value, "ok": value is not None}
            elif intent == "editor_lineNumbers":
                value = data.get("editor.lineNumbers")
                return {"intent": intent, "value": value, "ok": value is not None}
            elif intent == "exclude_pycache_exists":
                excludes = data.get("files.exclude")
                exists = False
                if isinstance(excludes, dict):
                    exists = (excludes.get("**/__pycache__") is True)
                details = "files.exclude contains '**/__pycache__': true" if exists else "Not found"
                return {"intent": intent, "value": exists, "ok": exists, "details": details}
            else:
                return self._extract_by_key_or_path(data, intent)
        except Exception as e:
            return {"intent": intent, "ok": False, "error": f"Extraction error: {str(e)}"}

    def forward(self, config_file_path: str, intent: str) -> Dict[str, Any]:
        try:
            data = self._load_json(config_file_path)
            if data is None:
                return {"intent": intent, "ok": False, "error": "Failed to read JSON or file not found"}
            if not isinstance(data, dict):
                return {"intent": intent, "ok": False, "error": "settings.json is not a JSON object"}
            return self._extract(data, intent)
        except Exception as e:
            return {"intent": intent, "ok": False, "error": f"Failed to parse settings: {str(e)}"}

    def __call__(self, config_file_path: str, intent: str) -> Dict[str, Any]:
        return self.forward(config_file_path=config_file_path, intent=intent)

    def to_code_prompt(self) -> str:
        return (
            f"Use {self.name} to read values by intent or raw key/path from a cached VS Code settings.json on the host.\n"
            "Call get_vscode_user_settings_file first, then:\n"
            "- get_vscode_setting_value_by_intent(config_file_path=\"/path/to/settings.json\", intent=\"auto_save\")\n"
            "- get_vscode_setting_value_by_intent(config_file_path=\"/path/to/settings.json\", intent=\"exclude_pycache_exists\")\n"
            "- get_vscode_setting_value_by_intent(config_file_path=\"/path/to/settings.json\", intent=\"workbench.colorTheme\")\n"
            "- get_vscode_setting_value_by_intent(config_file_path=\"/path/to/settings.json\", intent=\"python.analysis.diagnosticSeverityOverrides.reportMissingImports\")\n"
        )
