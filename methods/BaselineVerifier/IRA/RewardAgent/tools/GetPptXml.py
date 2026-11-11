from __future__ import annotations

import json
import os
import zipfile
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from smolagents import Tool

DEFAULT_NAMESPACES: Dict[str, str] = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def _parse_namespaces(namespaces_json: Optional[str]) -> Dict[str, str]:
    if not namespaces_json:
        return dict(DEFAULT_NAMESPACES)
    try:
        parsed = json.loads(namespaces_json)
        if not isinstance(parsed, dict):
            return dict(DEFAULT_NAMESPACES)
        merged = dict(DEFAULT_NAMESPACES)
        merged.update({str(k): str(v) for k, v in parsed.items()})
        return merged
    except Exception:
        return dict(DEFAULT_NAMESPACES)


def _qname_to_prefixed(qname: str, nsmap: Dict[str, str]) -> str:
    if not qname:
        return qname
    if qname.startswith("{"):
        try:
            uri, local = qname[1:].split("}", 1)
            for prefix, ns_uri in nsmap.items():
                if ns_uri == uri:
                    return f"{prefix}:{local}"
            return qname
        except Exception:
            return qname
    return qname


def _attrib_with_prefixes(el: ET.Element, nsmap: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        for k, v in el.attrib.items():
            out[_qname_to_prefixed(k, nsmap)] = v
    except Exception:
        pass
    return out


def _element_to_node(el: ET.Element, nsmap: Dict[str, str]) -> Dict[str, Any]:
    try:
        tag = _qname_to_prefixed(el.tag, nsmap)
    except Exception:
        tag = el.tag
    try:
        text = (el.text or "").strip()
    except Exception:
        text = None
    try:
        inner_xml = ET.tostring(el, encoding="unicode")
    except Exception:
        inner_xml = None
    return {
        "tag": tag,
        "attrib": _attrib_with_prefixes(el, nsmap),
        "text": text,
        "inner_xml": inner_xml,
    }


class GetPptXmlTool(Tool):

    name = "getpptxml"
    description = (
        """Low-level OOXML slide XML reader and XPath query executor. Use this tool for precise inspection of tags/attributes inside a specific slide XML (ppt/slides/slide{n}.xml), especially elements not exposed by python-pptx.

Use when you need exact XML details such as transitions/animations (p:transition/p:anim), extLst, custom namespaces, or raw attributes referenced in the slide XML.

Capabilities:
- Execute ElementTree-compatible XPath with default namespaces (p/a/r).
- Return structured nodes (tag, attrib, text, inner_xml) or the entire slide XML via "__RAW__".
- Deterministic, offline, suited for debugging and validating low-level XML content.

Limitations:
- Only reads one slide XML; does not access relationships (_rels), notes slides, charts (ppt/charts/*.xml), media, masters/layouts, or other parts.
- Result size limits: ~200 matched nodes and ~200k characters for raw XML; truncates if exceeded.
- No cross-part aggregation or high-level semantic judgments.

Inputs: file_path (Host path), slide_idx (0-based), xpath (XPath or "__RAW__").
Output: JSON with success, reason, slide_idx, (xpath), and data (nodes/count/truncated or xml/truncated).
Prefer getpptxml over checkpptfile for precise, low-level XML inspection within a slide. Do not use for cross-slide or cross-part checks."""
    )
    inputs = {
        "file_path": {
            "description": "Path to the .pptx file on the Host (absolute or relative).",
            "type": "string",
        },
        "slide_idx": {
            "description": "0-based slide index to read (maps to ppt/slides/slide{idx+1}.xml).",
            "type": "integer",
        },
        "xpath": {
            "description": "XPath expression to query (e.g., .//p:transition/p:dissolve), or \"__RAW__\" to return whole slide XML.",
            "type": "string",
        },
    }
    output_type = "string"

    def forward(
        self,
        file_path: str,
        slide_idx: int,
        xpath: str,
    ) -> str:
        return self.__call__(file_path, slide_idx, xpath)

    def __call__(
        self,
        file_path: str,
        slide_idx: int,
        xpath: str,
    ) -> str:
        try:
            if not file_path:
                return json.dumps({"success": False, "reason": "file_path is required"}, ensure_ascii=False)
            if slide_idx is None or not isinstance(slide_idx, int):
                return json.dumps({"success": False, "reason": "slide_idx must be an integer"}, ensure_ascii=False)

            abs_path = os.path.abspath(file_path)
            if not os.path.exists(abs_path):
                return json.dumps({
                    "success": False,
                    "reason": (
                        f"File not found on host: {abs_path}. "
                        "If your file is inside a VM (e.g., /home/user/...), download it to Host first and pass the Host path."
                    )
                }, ensure_ascii=False)

            part_name = f"ppt/slides/slide{slide_idx + 1}.xml"
            with zipfile.ZipFile(abs_path, "r") as zf:
                try:
                    zf.getinfo(part_name)
                except KeyError:
                    return json.dumps({
                        "success": False,
                        "reason": f"Slide XML not found: {part_name}",
                        "slide_idx": slide_idx,
                    }, ensure_ascii=False)
                with zf.open(part_name) as fh:
                    xml_bytes = fh.read()

            if xpath == "__RAW__":
                try:
                    xml_text = xml_bytes.decode("utf-8", errors="replace")
                except Exception:
                    xml_text = xml_bytes.decode("latin-1", errors="replace")
                truncated = False
                MAX_CHARS = 200000
                if len(xml_text) > MAX_CHARS:
                    xml_text = xml_text[:MAX_CHARS]
                    truncated = True
                return json.dumps({
                    "success": True,
                    "reason": "",
                    "slide_idx": slide_idx,
                    "data": {"xml": xml_text, "truncated": truncated},
                }, ensure_ascii=False)

            nsmap = dict(DEFAULT_NAMESPACES)
            try:
                root = ET.fromstring(xml_bytes)
            except Exception as e:
                return json.dumps({
                    "success": False,
                    "reason": f"Failed to parse slide XML: {e}",
                    "slide_idx": slide_idx,
                    "xpath": xpath,
                }, ensure_ascii=False)

            try:
                matches = root.findall(xpath, nsmap)
            except Exception as e:
                return json.dumps({
                    "success": False,
                    "reason": f"Invalid XPath or query error: {e}",
                    "slide_idx": slide_idx,
                    "xpath": xpath,
                }, ensure_ascii=False)

            nodes: List[Dict[str, Any]] = []
            truncated = False
            count = len(matches)
            limit = 200
            try:
                for i, el in enumerate(matches):
                    if i >= limit:
                        truncated = True
                        break
                    nodes.append(_element_to_node(el, nsmap))
            except Exception:
                truncated = truncated or (count > limit)

            return json.dumps({
                "success": True,
                "reason": "",
                "slide_idx": slide_idx,
                "xpath": xpath,
                "data": {"nodes": nodes, "count": count, "truncated": truncated},
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"success": False, "reason": f"Error: {e}"}, ensure_ascii=False)

    def to_code_prompt(self) -> str:
        return (
            "def getpptxml(file_path: str, slide_idx: int, xpath: str) -> str:\n"
            "    '''Read PPTX internal slide XML or run an XPath query and return structured results.\n"
            "    Pass xpath='__RAW__' to return entire slide XML.\n"
            "    Returns a JSON string with keys: success, reason, slide_idx, (xpath when provided), data.'''\n"
        )
