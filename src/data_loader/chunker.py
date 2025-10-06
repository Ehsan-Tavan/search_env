from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    def __init__(
            self,
            chunk_size: int = 500,
            chunk_overlap: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
            self,
            texts: List[str],
    ) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        all_chunks = []
        for doc in texts:
            chunks = splitter.split_text(doc)
            all_chunks.extend(chunks)
        return all_chunks
