from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str):
        print(f"🔹 Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts):
        return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
