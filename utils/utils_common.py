import re

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\x00", " ").replace("\u0000", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()
