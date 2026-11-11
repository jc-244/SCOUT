from typing import Any, Dict, List
from smolagents import Tool
import json


class GetVSCodeExtensionsTool(Tool):
    name = "get_vscode_extensions"
    description = (
        "Query the installed VS Code extensions via the VM using `code --list-extensions`. "
        "Returns a structured result with the parsed list and raw output. "
        "No insiders input; env is injected via constructor."
    )
    inputs = {}
    output_type = "object"

    def __init__(self, env: Any):
        super().__init__()
        self.env = env

    def _run_list_extensions(self) -> Dict[str, Any]:
        try:
            code = (
                'import subprocess, json\n'
                'p = subprocess.run("code --list-extensions", shell=True, capture_output=True, text=True)\n'
                'print(json.dumps({\n'
                '  "returncode": p.returncode,\n'
                '  "stdout": p.stdout,\n'
                '  "stderr": p.stderr,\n'
                '}))\n'
            )
            controller = getattr(self.env, "controller", None)
            if controller is None:
                return {"ok": False, "error": "Missing env.controller"}
            out_obj = controller.execute_python_command(code) if hasattr(controller, "execute_python_command") else None
            raw = (out_obj or {}).get("output", "")
            if not raw:
                return {"ok": False, "error": "VM command produced no output"}
            last_line = raw.strip().splitlines()[-1]
            try:
                data = json.loads(last_line)
            except Exception:
                parsed = [line.strip() for line in raw.splitlines() if line.strip()]
                return {"ok": True, "extensions": parsed, "raw_output": raw}
            rc = data.get("returncode", 0)
            stdout = data.get("stdout", "") or ""
            stderr = data.get("stderr", "") or ""
            parsed = [line.strip() for line in stdout.splitlines() if line.strip()]
            return {"ok": rc == 0, "extensions": parsed, "raw_output": stdout, "error": (stderr if rc != 0 else None)}
        except Exception as e:
            return {"ok": False, "error": f"Failed to run list-extensions: {str(e)}"}

    def forward(self) -> Dict[str, Any]:
        return self._run_list_extensions()

    def __call__(self) -> Dict[str, Any]:
        return self.forward()

    def to_code_prompt(self) -> str:
        return (
            f"Use {self.name} to retrieve installed VS Code extensions from the VM.\n"
            "Example:\n"
            "- get_vscode_extensions()  # returns {'ok': bool, 'extensions': [...], 'raw_output': '...', 'error'?: '...'}"
        )
