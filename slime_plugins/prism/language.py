"""Canonical language conditions shared with the Prism revision 7 renderer."""

# Keep these full names aligned with the pretrained processor's SUPPORTED_LANGUAGES.
# Language codes such as zh/en are WER metadata aliases, not renderer language names.
SUPPORTED_LANGUAGES = (
    "Chinese",  # 中文
    "Czech",  # 捷克语
    "Danish",  # 丹麦语
    "Dutch",  # 荷兰语
    "English",  # 英语
    "Finnish",  # 芬兰语
    "French",  # 法语
    "German",  # 德语
    "Greek",  # 希腊语
    "Hindi",  # 印地语
    "Hungarian",  # 匈牙利语
    "Indonesian",  # 印度尼西亚语
    "Italian",  # 意大利语
    "Japanese",  # 日语
    "Korean",  # 韩语
    "Malay",  # 马来语
    "Polish",  # 波兰语
    "Portuguese (Portugal)",  # 葡萄牙语（葡萄牙）
    "Romanian",  # 罗马尼亚语
    "Russian",  # 俄语
    "Spanish (Spain)",  # 西班牙语（西班牙）
    "Swedish",  # 瑞典语
    "Thai",  # 泰语
    "Turkish",  # 土耳其语
    "Vietnamese",  # 越南语
)
_LANGUAGE_NAMES = {name.casefold(): name for name in SUPPORTED_LANGUAGES}


def canonical_language(value):
    """Resolve a full language name without guessing from codes, text or domain."""
    name = _LANGUAGE_NAMES.get(value.strip().casefold()) if isinstance(value, str) else None
    if name is None:
        raise ValueError(
            "metadata.show_language=true requires metadata.language to be a supported Prism language name "
            f"(for example Chinese or English, not zh or en); got {value!r}"
        )
    return name
