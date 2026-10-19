import json
from pathlib import Path

import trafilatura
from bs4 import BeautifulSoup


RAW_DIR = Path("data/raw/html")
MANIFEST_FILE = Path("data/manifests/pages.jsonl")
OUTPUT_FILE = Path("data/processed/extracted_documents.jsonl")


def extract_text(html):
    """Extract the main readable text from an HTML document."""

    text = trafilatura.extract(
        html,
        include_tables=True,
        include_links=False,
        include_comments=False
    )

    # Fallback if the primary extractor returns no text.
    if not text or not text.strip():
        soup = BeautifulSoup(html, "lxml")

        for element in soup(
            ["script", "style", "nav", "footer", "header"]
        ):
            element.decompose()

        text = soup.get_text(
            separator="\n",
            strip=True
        )

    return text.strip() if text else ""


def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load the crawler's metadata.
    manifest = {}

    with MANIFEST_FILE.open(
        encoding="utf-8"
    ) as file:

        for line in file:
            record = json.loads(line)
            manifest[record["html_file"]] = record

    extracted_count = 0
    skipped_count = 0

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as output:

        for html_file in RAW_DIR.glob("*.html"):

            html = html_file.read_text(
                encoding="utf-8"
            )

            text = extract_text(html)

            if not text:
                print("No text extracted:", html_file)
                skipped_count += 1
                continue

            metadata = manifest.get(
                str(html_file),
                {}
            )

            document = {
                "source_url": metadata.get("url", ""),
                "title": metadata.get("title", ""),
                "text": text,
                "source_file": str(html_file),
                "word_count": len(text.split())
            }

            output.write(
                json.dumps(
                    document,
                    ensure_ascii=False
                ) + "\n"
            )

            extracted_count += 1

            print(
                f"Extracted: {document['title']} "
                f"({document['word_count']} words)"
            )

    print("\nExtraction complete.")
    print("Documents extracted:", extracted_count)
    print("Documents skipped:", skipped_count)
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    main()