from tqdm import tqdm
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

    def chunk_text(self, texts, batch_size: int = 256):
        all_chunks = []

        # Process texts in batches for faster tokenization
        for i in tqdm(range(0, len(texts), batch_size), desc="Chunking texts", ncols=80):
            batch = texts[i:i + batch_size]

            # Batch encode to reduce overhead
            encoded = self.tokenizer(batch, add_special_tokens=False, truncation=False)

            for tokens in encoded["input_ids"]:
                step = self.chunk_size - self.chunk_overlap

                # Slice token sequence into chunks
                for j in range(0, len(tokens), step):
                    chunk_tokens = tokens[j:j + self.chunk_size]
                    # Fast decode without cleaning up extra spaces
                    chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                    all_chunks.append(chunk_text.strip())

        return all_chunks
