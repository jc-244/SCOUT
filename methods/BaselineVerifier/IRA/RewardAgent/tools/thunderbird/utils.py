import json
import re
from typing import Any, Dict, Optional

_PREF_LINE_RE = re.compile(r'user_pref\("([^"]+)",\s*(.+?)\);\s*$')


def _parse_value(raw: str) -> Any:
    s = raw.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        inner = s[1:-1]
        inner = inner.replace('\\"', '"')
        inner = inner.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
        return inner
    if s.lower() == 'true':
        return True
    if s.lower() == 'false':
        return False
    try:
        return int(s)
    except ValueError:
        pass
    return s


def load_prefs_map(config_file_path: str) -> Dict[str, Any]:
    prefs: Dict[str, Any] = {}
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                m = _PREF_LINE_RE.search(line)
                if not m:
                    continue
                key = m.group(1)
                raw_val = m.group(2)
                prefs[key] = _parse_value(raw_val)
    except Exception:
        return prefs
    return prefs


def get_pref_value(prefs: Dict[str, Any], key: str) -> Optional[Any]:
    return prefs.get(key)


def to_json_string(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)
