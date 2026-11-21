import re

_STRIP = re.compile("[\u064B-\u0652\u0670\u0640]")
_TABLE = str.maketrans(
    {
        "\u0623": "\u0627",  # أ
        "\u0625": "\u0627",  # إ
        "\u0622": "\u0627",  # آ
        "\u0671": "\u0627",  # ٱ
        "\u0649": "\u064A",  # ى -> ي
        "\u0629": "\u0647",  # ة -> ه
        **{chr(0x0660 + i): str(i) for i in range(10)},  # Arabic-Indic digits
        **{chr(0x06F0 + i): str(i) for i in range(10)},  # Persian digits
    }
)
_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+")


def normalize_ar(text: str) -> str:
    text = _STRIP.sub("", text)
    text = text.translate(_TABLE).lower()
    return _WS.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return _WORD.findall(normalize_ar(text))