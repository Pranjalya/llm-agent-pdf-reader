import logging
from typing import List

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import DocChunk
from docling_core.types.doc.document import DoclingDocument, TextItem

from src.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class PDFProcessor:
    """Handles PDF document processing."""
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.text_content: DoclingDocument | None = None
        self.chunker = None
        self.chunks: List[DocChunk] | None = None

    def load_and_chunk(self) -> List[str]:
        """Loads the PDF and chunks the text."""
        self._extract_text()
        if self.text_content:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL_NAME)
            self.chunker = HybridChunker(tokenizer=tokenizer)
            return self.chunk_text()
        return []

    def _extract_text(self) -> None:
        """Extracts text content from the PDF."""
        try:
            conv_res = DocumentConverter().convert(self.pdf_path)
            self.text_content = conv_res.document
            logger.info(f"Successfully extracted text from {self.pdf_path}")
        except FileNotFoundError:
            logger.error(f"PDF file not found at: {self.pdf_path}")
            raise
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

    def chunk_text(self) -> List[str]:
        """Splits the text into smaller chunks."""
        if self.chunker and self.text_content:
            chunk_iter = self.chunker.chunk(dl_doc=self.text_content)
            self.chunks = list(chunk_iter)
            return list(map(self.get_chunk_with_title, self.chunks))
        return []

    def get_origin_text_from_chunk(self, chunk: DocChunk) -> str:
        """Get origin text from the chunk. Returns chunked text if no origin text available."""
        text = " ".join([doc_item.orig for doc_item in chunk.meta.doc_items if type(doc_item) is TextItem])
        if len(text.strip()) == 0:
            text = chunk.text
        return text

    def get_chunk_with_title(self, chunk: DocChunk) -> str:
        """Returns formatted chunk with section heading"""
        return f"Title : {' '.join(chunk.meta.headings)} | {chunk.text}".replace("\n", " ")

    def get_chunk(self, index: int) -> DocChunk:
        """Returns chunk by index"""
        return self.chunks[index]