import json
from typing import Any, Dict, List, Optional
from smolagents import Tool


class CheckVSCodeKeybindingByIntentTool(Tool):
    name = "check_vscode_keybinding_by_intent"
    description = (
        "Host-only check for specific VS Code keybindings by intent (keyless). "
        "Loads a cached keybindings.json and verifies expected entries are present. "
        "Intents supported: remove_tree_view_find, focus_editor_from_terminal."
    )
    inputs = {
        "config_file_path": {
            "type": "string",
            "description": "Host path to cached VS Code keybindings.json (from get_vscode_keybindings_file)",
        },
        "intent": {
            "type": "string",
            "description": "One of: 'remove_tree_view_find', 'focus_editor_from_terminal'",
        },
    }
    output_type = "object"

    def __init__(self):
        super().__init__()
        self.env = None

    def _load_json_list(self, path: str) -> Optional[List[Dict[str, Any]]]:
        for mode in ("direct", "skip_first_line"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    if mode == "skip_first_line":
                        f.readline()
                    data = json.load(f)
                if isinstance(data, list):
                    return data
            except Exception:
                pass
        return None

    def _expected_for_intent(self, intent: str) -> Optional[Dict[str, Any]]:
        if intent == "remove_tree_view_find":
            return {
                "key": "ctrl+f",
                "command": "-list.find",
                "when": "listFocus && listSupportsFind",
            }
        elif intent == "focus_editor_from_terminal":
            return {
                "key": "ctrl+j",
                "command": "workbench.action.focusActiveEditorGroup",
                "when": "terminalFocus",
            }
        return None

    def forward(self, config_file_path: str, intent: str) -> Dict[str, Any]:
        expected = self._expected_for_intent(intent)
        if expected is None:
            return {"intent": intent, "ok": False, "error": f"Unsupported intent: {intent}"}
        data = self._load_json_list(config_file_path)
        if data is None:
            return {"intent": intent, "ok": False, "error": "Failed to read keybindings.json or not a list"}
        for idx, item in enumerate(data):
            if item == expected:
                return {
                    "intent": intent,
                    "ok": True,
                    "found_entry": expected,
                    "index": idx,
                    "match_strategy": "exact_dict_in_list",
                }
        return {
            "intent": intent,
            "ok": False,
            "details": "Expected keybinding not found",
            "expected": expected,
        }

    def __call__(self, config_file_path: str, intent: str) -> Dict[str, Any]:
        return self.forward(config_file_path=config_file_path, intent=intent)

    def to_code_prompt(self) -> str:
        return (
            f"Use {self.name} to verify keybindings by intent from cached keybindings.json.\n"
            "First call get_vscode_keybindings_file to cache the file, then:\n"
            "- check_vscode_keybinding_by_intent(config_file_path=\"/path/keybindings.json\", intent=\"remove_tree_view_find\")\n"
            "- check_vscode_keybinding_by_intent(config_file_path=\"/path/keybindings.json\", intent=\"focus_editor_from_terminal\")\n"
        )
