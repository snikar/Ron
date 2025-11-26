"""
Parsers Module (Phase 1)
------------------------
Handles file ingestion for Jeff:
 - PDF
 - Images (OCR)
 - DOCX
 - TXT / Markdown
 - CSV
 - Excel

Each parser returns CLEAN TEXT.
Chunking + embedding are handled by MemoryManager + EmbeddingEngine.

Requirements:
 - No circular imports
 - Portable
 - Minimal dependencies
"""

import io
import pytesseract
import pandas as pd
from PIL import Image
from pdfminer.high_level import extract_text
from docx import Document


class ParserEngine:
    """
    Main entry point for all file → text conversion.
    """

    def __init__(self):
        pass

    # ---------------------------------------------------------
    #                     MASTER PARSER
    # ---------------------------------------------------------

    def parse_file(self, file_bytes: bytes, filename: str) -> str:
        """
        Detect file type from extension and route accordingly.
        Returns plain text.
        """

        ext = filename.lower()

        if ext.endswith(".pdf"):
            return self._parse_pdf(file_bytes)

        if ext.endswith(".png") or ext.endswith(".jpg") or ext.endswith(".jpeg"):
            return self._parse_image(file_bytes)

        if ext.endswith(".docx"):
            return self._parse_docx(file_bytes)

        if ext.endswith(".txt") or ext.endswith(".md"):
            return self._parse_text(file_bytes)

        if ext.endswith(".csv"):
            return self._parse_csv(file_bytes)

        if ext.endswith(".xlsx") or ext.endswith(".xls"):
            return self._parse_excel(file_bytes)

        return f"[Unsupported file type: {filename}]"

    # ---------------------------------------------------------
    #                       PDF PARSER
    # ---------------------------------------------------------

    def _parse_pdf(self, file_bytes: bytes) -> str:
        try:
            # pdfminer wants a file path or a file-like object
            text = extract_text(io.BytesIO(file_bytes))
            return self._clean(text)
        except Exception as e:
            return f"[PDF Parse Error] {e}"

    # ---------------------------------------------------------
    #                     IMAGE → OCR
    # ---------------------------------------------------------

    def _parse_image(self, file_bytes: bytes) -> str:
        try:
            img = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(img)
            return self._clean(text)
        except Exception as e:
            return f"[Image OCR Error] {e}"

    # ---------------------------------------------------------
    #                      DOCX PARSER
    # ---------------------------------------------------------

    def _parse_docx(self, file_bytes: bytes) -> str:
        try:
            file_obj = io.BytesIO(file_bytes)
            doc = Document(file_obj)
            text = "\n".join([para.text for para in doc.paragraphs])
            return self._clean(text)
        except Exception as e:
            return f"[DOCX Parse Error] {e}"

    # ---------------------------------------------------------
    #                  TEXT / MARKDOWN
    # ---------------------------------------------------------

    def _parse_text(self, file_bytes: bytes) -> str:
        try:
            return self._clean(file_bytes.decode("utf-8", errors="ignore"))
        except Exception as e:
            return f"[Text Parse Error] {e}"

    # ---------------------------------------------------------
    #                      CSV PARSER
    # ---------------------------------------------------------

    def _parse_csv(self, file_bytes: bytes) -> str:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
            return self._clean(df.to_string())
        except Exception as e:
            return f"[CSV Parse Error] {e}"

    # ---------------------------------------------------------
    #                      EXCEL PARSER
    # ---------------------------------------------------------

    def _parse_excel(self, file_bytes: bytes) -> str:
        try:
            df = pd.read_excel(io.BytesIO(file_bytes))
            return self._clean(df.to_string())
        except Exception as e:
            return f"[Excel Parse Error] {e}"

    # ---------------------------------------------------------
    #                     CLEANUP
    # ---------------------------------------------------------

    def _clean(self, text: str) -> str:
        """
        Normalize whitespace.
        """

        if not text:
            return ""

        # Collapse multiple whitespace
        cleaned = " ".join(text.split())
        return cleaned.strip()
