from io import text_encoding
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union
import docx
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

def extract_text_from_pdf(file_path: Path) -> List[tuple[int,str]]:
    """Extract text from a PDF file per page. Returns list of (page_number, text)."""
    readeer= PdfReader(str(file_path))
    pages=[]
    for idx, page in enumerate(readeer.pages):
        text  = page.extract_text() or ""
        if text.strip():
            pages.append((idx+1, text))
    
    return pages

import re


def extract_text_from_docx(file_path: Path) -> str:
    """Extract text from a DOCX or fallback DOC/RTF file."""
    try:
        doc = docx.Document(str(file_path))
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
        if text.strip():
            return text
    except Exception:
        pass

    # Fallback for RTF or text-based .doc files
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        raw_content = f.read()

    if raw_content.startswith("{\\rtf") or "\\par" in raw_content:
        text = re.sub(r"\\par[d]?", "\n", raw_content)
        text = re.sub(r"\\[a-zA-Z0-9\-]+", " ", text)
        text = re.sub(r"[{}]", "", text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n\n".join(lines)

    return raw_content


def extract_text_from_txt(file_path: Path) -> str:
    """Extract text from a plain text or Markdown file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_and_chunk(
    file_path: Union[str, Path],
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP
) -> List[Chunk]:
    """Load a document (PDF, DOCX, TXT, MD) and split it into uniform chunks."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks: List[Chunk] = []
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        pages = extract_text_from_pdf(path)
        for page_num, page_text in pages:
            page_chunks = splitter.split_text(page_text)
            for chunk_idx, text in enumerate(page_chunks):
                chunks.append(
                    Chunk(
                        content=text,
                        metadata={
                            "source": str(path.name),
                            "page": page_num,
                            "chunk_index": chunk_idx
                        }
                    )
                )
    
    elif suffix in [".docx", ".doc"]:
        full_text = extract_text_from_docx(path)
        text_chunks = splitter.split_text(full_text)
        for chunk_idx, text in enumerate(text_chunks):
            chunks.append(
                Chunk(
                    content=text,
                    metadata={
                        "source": str(path.name),
                        "chunk_index": chunk_idx
                    }
                )
            )

    elif suffix in [".txt", ".md"]:
        full_text = extract_text_from_txt(path)
        text_chunks = splitter.split_text(full_text)
        for chunk_idx, text in enumerate(text_chunks):
            chunks.append(
                Chunk(
                    content=text,
                    metadata={
                        "source": str(path.name),
                        "chunk_index": chunk_idx,
                    }
                )
            )
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    return chunks


if __name__ == "__main__":
    docs_dir = Path(settings.DATA_DIR)
    if docs_dir.exists():
        files = [
            f for f in docs_dir.iterdir()
            if f.is_file() and f.suffix.lower() in [".pdf", ".docx", ".txt", ".md"]
        ]
        if not files:
            print(f"No documents found in {docs_dir}. Place PDF/DOCX/TXT files there to test.")
        for file in files:
            result = load_and_chunk(file)
            print(f"📄 File: {file.name} -> Generated {len(result)} chunks.")
            if result:
                print(f"   First chunk metadata: {result[0].metadata}")
                print(f"   First chunk snippet:  {result[0].content[:120]}...\n")