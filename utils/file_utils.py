from pathlib import Path
import logging
from typing import Optional
from utils.text_utils import clean_text

def read_txt(path: Path) -> str:
    """Reads a text file and returns its content as a string."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logging.error(f"Error reading TXT file {path}: {e}")
        return ""

def read_pdf(path: Path) -> str:
    """Extracts text from a PDF file."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception as inner_e:
                logging.warning(f"Failed to extract text from a page in {path}: {inner_e}")
                pages.append("")
        return "\n".join(pages)
    except ImportError:
        logging.error("pypdf is not installed.")
        raise
    except Exception as e:
        logging.error(f"Error reading PDF file {path}: {e}")
        return ""

def read_epub(path: Path) -> str:
    """Extracts text from an EPUB file."""
    try:
        from ebooklib import epub
        from bs4 import BeautifulSoup
        book = epub.read_epub(str(path))
        texts = []
        for item in book.get_items():
            if item.get_type() == epub.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "lxml")
                texts.append(soup.get_text(" ", strip=True))
        return "\n".join(texts)
    except ImportError:
        logging.error("ebooklib or beautifulsoup4 is not installed.")
        raise
    except Exception as e:
        logging.error(f"Error reading EPUB file {path}: {e}")
        return ""

def load_book_text(path: Path, lang: str = "en") -> str:
    """Loads a book (TXT, PDF, EPUB) and cleans its text."""
    ext = path.suffix.lower()
    if ext in [".txt", ".md"]:
        raw = read_txt(path)
    elif ext == ".pdf":
        raw = read_pdf(path)
    elif ext == ".epub":
        raw = read_epub(path)
    else:
        raise ValueError(f"Unsupported file type: {ext} (use TXT/PDF/EPUB)")
    
    if not raw.strip():
        raise ValueError(f"No text could be extracted from the file: {path.name}")
        
    return clean_text(raw, lang=lang)
