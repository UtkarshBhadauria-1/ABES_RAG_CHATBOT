import json
import re
import hashlib
import unicodedata

from pathlib import Path
from urllib.parse import urlparse


# ----------------------------------
# 1. File paths
# ----------------------------------

INPUT_FILE = Path(
    "data/processed/extracted_documents.jsonl"
)

OUTPUT_FILE = Path(
    "data/processed/cleaned_documents.jsonl"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ----------------------------------
# 2. Text cleaning
# ----------------------------------

def clean_text(text):
    """
    Normalize Unicode and whitespace
    while preserving meaningful content.
    """

    # Standardize equivalent Unicode representations.
    text = unicodedata.normalize("NFKC", text)

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove unnecessary horizontal whitespace.
    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.split("\n")
    ]

    # Collapse consecutive blank lines.
    cleaned_lines = []

    for line in lines:
        if line:
            cleaned_lines.append(line)
        elif cleaned_lines and cleaned_lines[-1] != "":
            cleaned_lines.append("")

    return "\n".join(cleaned_lines).strip()


# ----------------------------------
# 3. Metadata validation
# ----------------------------------

def valid_source_url(url):
    """Check whether the source is an HTTPS URL."""

    if not isinstance(url, str) or not url.strip():
        return False

    parsed = urlparse(url)

    return (
        parsed.scheme == "https"
        and parsed.hostname == "www.abes.ac.in"
    )


# ----------------------------------
# 4. Preprocessing pipeline
# ----------------------------------

def preprocess():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}. "
            "Run src/extract.py first."
        )

    seen_content_hashes = set()

    total_records = 0
    cleaned_records = 0
    duplicate_records = 0
    invalid_records = 0
    empty_records = 0

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as source, OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as target:

        for line in source:

            total_records += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_records += 1
                continue

            text = record.get("text", "")

            if not isinstance(text, str):
                invalid_records += 1
                continue

            text = clean_text(text)

            if not text:
                empty_records += 1
                continue

            source_url = record.get(
                "source_url", ""
            )

            if not valid_source_url(source_url):
                invalid_records += 1
                continue

            title = record.get("title", "")

            if not isinstance(title, str) or not title.strip():
                title = "Untitled"

            # Hash cleaned text to identify exact duplicates.
            content_hash = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()

            if content_hash in seen_content_hashes:
                duplicate_records += 1
                continue

            seen_content_hashes.add(content_hash)

            record["title"] = title.strip()
            record["text"] = text
            record["word_count"] = len(text.split())
            record["character_count"] = len(text)
            record["content_hash"] = content_hash

            target.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

            cleaned_records += 1

    print("\nPreprocessing complete.")
    print("-----------------------")
    print("Input records:", total_records)
    print("Cleaned records:", cleaned_records)
    print("Duplicate records removed:", duplicate_records)
    print("Empty records skipped:", empty_records)
    print("Invalid records skipped:", invalid_records)
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    preprocess()