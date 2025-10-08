import re
from html import unescape


def clean_text(text: str) -> str:
    # 1. Unescape any HTML entities (e.g., &nbsp;, &amp;)
    text = unescape(text)

    # 2. Remove HTML tags and attributes like <span style="color: ...">
    text = re.sub(r'<[^>]+>', ' ', text)

    # 3. Remove attribute-like leftovers such as ="color: black">
    text = re.sub(r'=\"[^"]*\"', ' ', text)

    # 4. Remove any stray special characters left from markup
    text = re.sub(r'[<>]', ' ', text)

    # 5. Replace multiple spaces or line breaks with a single space
    text = re.sub(r'\s+', ' ', text)

    # 6. Replace Arabic letters with Persian equivalents
    text = text.replace("ي", "ی").replace("ك", "ک")

    # 6. Strip leading/trailing spaces
    text = text.strip()

    return text
