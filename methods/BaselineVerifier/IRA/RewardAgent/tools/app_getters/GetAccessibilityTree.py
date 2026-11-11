from smolagents import Tool
from desktop_env.evaluators.getters.misc import get_accessibility_tree
import re


def optimize_accessibility_xml(xml_content: str) -> str:
    if not xml_content or len(xml_content) < 1000:
        return xml_content

    original_len = len(xml_content)

    xml_content = re.sub(r'\s+act:[a-zA-Z_]+=\"[^\"]*\"', '', xml_content)

    default_true_attrs = [
        'st:visible',
        'st:enabled',
        'st:sensitive',
        'st:focusable',
        'st:showing',
        'st:selectable_text',
    ]
    for attr in default_true_attrs:
        xml_content = re.sub(rf'\s+{attr}=\"true\"', '', xml_content)

    xml_content = re.sub(r'\s+[a-zA-Z_]+:[a-zA-Z_]+=\"\"', '', xml_content)


    xml_content = re.sub(r'\s+>', '>', xml_content)
    xml_content = re.sub(r'\s+\s+', ' ', xml_content)

    optimized_len = len(xml_content)
    reduction = (original_len - optimized_len) / original_len * 100

    if reduction > 10:
        print(f"[GetAccessibilityTree] Optimized XML: {original_len} -> {optimized_len} chars ({reduction:.1f}% reduction)")

    return xml_content


class GetAccessibilityTreeTool(Tool):
    name = "get_accessibility_tree"
    description = (
        "Fetch the current Accessibility Tree (A11y) XML from the active desktop environment.\n\n"
        "This returns an optimized XML string (reduced ~58% by removing redundant attributes).\n"
        "The optimization removes: empty action attributes, default true state values, and action descriptions.\n"
        "You can parse it using lxml with the provided namespaces."
    )
    inputs = {}
    output_type = "string"

    def __init__(self, env):
        super().__init__()
        self.env = env

    def forward(self) -> str:
        try:
            at_xml: str = get_accessibility_tree(self.env)
            if at_xml is None:
                return ""

            optimized_xml = optimize_accessibility_xml(at_xml)
            return optimized_xml
        except Exception as e:
            return f"Error: Failed to get accessibility tree: {str(e)}"

    def __call__(self) -> str:
        return self.forward()

    def to_code_prompt(self) -> str:
        return (
            "Use get_accessibility_tree() to retrieve the desktop Accessibility Tree (XML).\n"
            "Example:\n"
            "- xml_str = get_accessibility_tree()\n\n"
            "Notes:\n"
            "- Output is an optimized XML string (~58% smaller than raw output).\n"
            "- Namespace map differs by OS but is provided inside the XML's nsmap.\n"
        )
