from .CaptionImage import CaptionImageTool
from .CheckExcelFile import CheckExcelFileTool
from .CheckPptFile import CheckPptFileTool
from .CheckWordFile import CheckWordFileTool
from .GetPptXml import GetPptXmlTool

try:
    from .thunderbird import (
        GetThunderbirdPrefsFileTool,
        GetThunderbirdActiveThemeTool,
        GetThunderbirdRegionTool,
        GetThunderbirdTimezoneTool,
        GetThunderbirdUseSystemTimezoneTool,
        GetThunderbirdAccountsTool,
        GetThunderbirdImapServerInfoTool,
        GetThunderbirdSmtpServerInfoTool,
    )
    _THUNDERBIRD_AVAILABLE = True
except Exception as _th_e:
    _THUNDERBIRD_AVAILABLE = False


from .vscode_getters import (
    GetVSCodeUserSettingsFileTool,
    GetVSCodeSettingValueByIntentTool,
    GetVSCodeExtensionsTool,
    GetVSCodeKeybindingsFileTool,
    CheckVSCodeKeybindingByIntentTool,
    GetVSCodeWorkspaceSettingValueByIntentTool,
)

__all__ = [
    'CaptionImageTool',
    'CheckExcelFileTool',
    'CheckPptFileTool',
    'CheckWordFileTool',
    'GetPptXmlTool',
    'GetVSCodeUserSettingsFileTool',
    'GetVSCodeSettingValueByIntentTool',
    'GetVSCodeExtensionsTool',
    'GetVSCodeKeybindingsFileTool',
    'CheckVSCodeKeybindingByIntentTool',
    'GetVSCodeWorkspaceSettingValueByIntentTool',
]

if _THUNDERBIRD_AVAILABLE:
    __all__.extend([
        'GetThunderbirdPrefsFileTool',
        'GetThunderbirdActiveThemeTool',
        'GetThunderbirdRegionTool',
        'GetThunderbirdTimezoneTool',
        'GetThunderbirdUseSystemTimezoneTool',
        'GetThunderbirdAccountsTool',
        'GetThunderbirdImapServerInfoTool',
        'GetThunderbirdSmtpServerInfoTool',
    ])
