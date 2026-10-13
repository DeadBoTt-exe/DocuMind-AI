"""Text cleaning: removes boilerplate and formatting artifacts from raw document text."""

import re

AWS_HEADER_PATTERNS = [
    r"AWS Organizations User Guide",
    r"©\s*Amazon Web Services",
]


def clean_text(text: str) -> str:
    for pattern in AWS_HEADER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\n\d+\n", "\n", text)   # page numbers first...
    text = re.sub(r"\n{3,}", "\n\n", text)   # ...then collapse the blank lines they leave
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()
