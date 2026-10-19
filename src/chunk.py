import json
import hashlib

from pathlib import Path


# ----------------------------------
# 1. Configuration
# ----------------------------------

INPUT_FILE = Path(
    "data/processed/cleaned_documents.jsonl"
)

OUTPUT_FILE = Path(
    "data/processed/chunks.jsonl"
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# ----------------------------------
# 2. Create chunks
# ----------------------------------

def create_chunks(text, chunk_size=500, overlap=100):
    """
    Split text into word-based chunks.

    Each chunk contains at most chunk_size words.
    Consecutive chunks share `overlap` words.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and < chunk_size"
        )

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words)
        )

        chunk_text = " ".join(words[start:end])

        chunks.append({
            "text": chunk_text,
            "start_word": start,
            "end_word": end
        })

        if end == len(words):
            break

        start = end - overlap

    return chunks


# ----------------------------------
# 3. Process documents
# ----------------------------------

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}. "
            "Run src/preprocess.py first."
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    total_documents = 0
    total_chunks = 0

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as source, OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as target:

        for line in source:

            document = json.loads(line)

            total_documents += 1

            chunks = create_chunks(
                document["text"],
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP
            )

            for index, chunk in enumerate(chunks):

                chunk_id_source = (
                    f"{document['source_url']}"
                    f"|{index}|{chunk['text']}"
                )

                chunk_id = hashlib.sha256(
                    chunk_id_source.encode("utf-8")
                ).hexdigest()

                record = {
                    "chunk_id": chunk_id,
                    "source_url": document["source_url"],
                    "title": document["title"],
                    "chunk_index": index,
                    "start_word": chunk["start_word"],
                    "end_word": chunk["end_word"],
                    "text": chunk["text"]
                }

                target.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )

                total_chunks += 1

    print("\nChunking complete.")
    print("-------------------")
    print("Documents processed:", total_documents)
    print("Total chunks created:", total_chunks)
    print("Chunk size:", CHUNK_SIZE, "words")
    print("Chunk overlap:", CHUNK_OVERLAP, "words")
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    main()