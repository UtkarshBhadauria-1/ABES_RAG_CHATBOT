import json
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer


# ----------------------------------
# 1. Configuration
# ----------------------------------

INPUT_FILE = Path(
    "data/processed/chunks.jsonl"
)

OUTPUT_FILE = Path(
    "data/processed/chunk_embeddings.npz"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ----------------------------------
# 2. Load chunks
# ----------------------------------

def load_chunks():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {INPUT_FILE}. "
            "Run src/chunk.py first."
        )

    chunks = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if line.strip():
                chunks.append(json.loads(line))

    if not chunks:
        raise ValueError(
            "The chunk dataset is empty."
        )

    return chunks


# ----------------------------------
# 3. Generate embeddings
# ----------------------------------

def main():

    chunks = load_chunks()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("Chunks loaded:", len(texts))
    print("Loading embedding model:", MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    # Preserve metadata in the same order as the vectors.
    metadata = [
        {
            "chunk_id": chunk["chunk_id"],
            "source_url": chunk["source_url"],
            "title": chunk["title"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"]
        }
        for chunk in chunks
    ]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_FILE,
        embeddings=embeddings,
        metadata=np.array(
            [
                json.dumps(item, ensure_ascii=False)
                for item in metadata
            ]
        )
    )

    print("\nEmbedding generation complete.")
    print("Embedding shape:", embeddings.shape)
    print("Embedding dimensions:", embeddings.shape[1])
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    main()