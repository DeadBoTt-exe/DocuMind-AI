"""Text cleaning: removes boilerplate and formatting artifacts from raw document text."""

import re

# \s+ (not a literal space): in the real PDF the running header is split across
# lines ("AWS Organizations\nUser Guide"), which a single-space pattern never matches.
AWS_HEADER_PATTERNS = [
    r"AWS\s+Organizations\s*:?\s*User\s+Guide",
    r"©\s*(?:\d{4}\s+)?Amazon\s+Web\s+Services",
    # The tag-policy table's column headers are repeated on hundreds of pages and
    # extract as scrambled fragments ("CloudForm ation", "Enforceme nt Mode").
    # Strip only this exact block; do NOT drop chunks heuristically, because lists
    # of service names and JSON examples look "noisy" but are real content.
    (
        r"Basic\s+Compliance\s+Rules\s+Required\s+Tag\s+Keys\s+Service\s+Tag\s+Policies"
        r"\s+JSON\s+syntax\s+CloudForm\s*ation\s+Alias\s+Reporting\s+Mode"
        r"\s+Enforceme\s*nt\s+Mode\s+Reporting\s+Mode\s+Enforce\s+for\s+IaC"
    ),
]


def clean_text(text: str) -> str:
    for pattern in AWS_HEADER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\n\d+\n", "\n", text)   # page numbers first...
    text = re.sub(r"\n{3,}", "\n\n", text)   # ...then collapse the blank lines they leave
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()
