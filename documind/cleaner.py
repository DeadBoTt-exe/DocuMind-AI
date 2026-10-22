"""Text cleaning: removes boilerplate and formatting artifacts from raw document text."""

import re
import unicodedata

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
    """Clean text by stripping headers, footers, and page numbers safely.
    
    Page numbers are only stripped at boundaries (leading or trailing page numbers),
    never in the middle of sentences or lists (e.g. numbered steps '1', '2' or table values).
    """
    text = unicodedata.normalize("NFKC", text)

    for pattern in AWS_HEADER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Strip standalone page numbers only at the start or end of the page/text
    text = re.sub(r"^\s*\d+\s*\n", "", text)
    text = re.sub(r"\n\s*\d+\s*$", "", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()
