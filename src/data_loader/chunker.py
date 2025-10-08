from transformers import AutoTokenizer


class Chunker:
    def __init__(
            self,
            chunk_size: int = 500,
            chunk_overlap: int = 100,
            model_name: str = None,
    ):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def chunk_text(self, texts):
        all_chunks = []
        for doc in texts:
            # Encode into tokens
            tokens = self.tokenizer.encode(doc, add_special_tokens=False)

            # Chunk by token length
            for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
                chunk_tokens = tokens[i:i + self.chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                all_chunks.append(chunk_text.strip())

        return all_chunks
