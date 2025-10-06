import torch
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str):
        self.device = self._detect_device()
        print(f"Loading model '{model_name}' on device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU detected: {gpu_name} ({num_gpus} GPU(s) available)")
            return "cuda"
        else:
            print("No GPU detected. Using CPU.")
            return "cpu"

    def encode_texts(
            self,
            texts,
            batch_size: int = 32
    ):
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                device=self.device
            )
            return embeddings
        except Exception as e:
            print(f"Error during embedding generation: {e}")
            raise
