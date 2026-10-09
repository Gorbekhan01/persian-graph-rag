import re
from parsivar import Normalizer

_NORMALIZER = Normalizer(
    date_normalizing_needed=False,
    statistical_space_correction=False
)


def clean_persian_text(text: str) -> str:
    # Clean and normalize Persian text 

    if not text:
        return ""

    # Remove unwanted special unicode characters
    text = text.replace("\ufeff", "")  # Byte Order Mark (BOM)
    text = text.replace("\u200a", " ")  # Hair Space
    text = text.replace("\u200f", "")  # Right-to-Left Mark (RLM)
    text = text.replace("\u200e", "")  # Left-to-Right Mark (LRM)


    # Collapse multiple whitespaces into a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse multiple newlines into max 2 newlines (preserves paragraph structure for RAG chunking)
    text = re.sub(r"\n\s*\n", "\n\n", text)

    return text.strip()


def normalize_query(query: str) -> str:
    # Normalize Persian user queries using Parsivar library.

    if not query:
        return ""
        
    cleaned = clean_persian_text(query)
    return _NORMALIZER.normalize(cleaned)
