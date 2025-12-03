import pickle
from pathlib import Path

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"
INDEX_PATH = BASE_DIR / "kb_index.pkl"


def extract_text_from_pdfs(kb_dir: Path):
    """Read all PDFs and return a list of {'source', 'page', 'text'} chunks."""
    chunks = []

    for pdf_path in kb_dir.glob("*.pdf"):
        reader = PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages):
            try:
                raw_text = page.extract_text() or ""
            except Exception:
                raw_text = ""

            text = " ".join(raw_text.split())
            if not text:
                continue

            # Break long pages into smaller chunks
            max_len = 900  # characters
            for start in range(0, len(text), max_len):
                snippet = text[start:start + max_len]
                chunks.append(
                    {
                        "source": pdf_path.name,
                        "page": page_num + 1,
                        "text": snippet,
                    }
                )

    return chunks


def build_index():
    if not KB_DIR.exists():
        raise FileNotFoundError(f"knowledge_base folder not found at {KB_DIR}")

    print("Reading PDFs...")
    chunks = extract_text_from_pdfs(KB_DIR)
    texts = [c["text"] for c in chunks]

    print(f"Loaded {len(chunks)} text chunks from PDFs.")

    print("Building TF-IDF index...")
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)

    data = {
        "chunks": chunks,
        "vectorizer": vectorizer,
        "matrix": matrix,
    }

    with open(INDEX_PATH, "wb") as f:
        pickle.dump(data, f)

    print(f"Knowledge base index saved to {INDEX_PATH}")


if __name__ == "__main__":
    build_index()
