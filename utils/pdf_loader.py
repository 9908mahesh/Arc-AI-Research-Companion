from pypdf import PdfReader
from langchain.docstore.document import Document
from .text_splitter import chunk_document_text
from .utils_common import clean_text
import os

def load_pdf_as_documents(file_path, source_name=None):
    """
    Return list of langchain Documents with metadata: {'source': source_name, 'page': page_num}
    """
    if source_name is None:
        source_name = os.path.basename(file_path)
    reader = PdfReader(file_path)
    docs = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = clean_text(text)
        if not text:
            continue
        # chunk this page's text into smaller chunks
        chunks = chunk_document_text(text)
        for chunk in chunks:
            metadata = {"source": source_name, "page": i}
            docs.append(Document(page_content=chunk, metadata=metadata))
    return docs
