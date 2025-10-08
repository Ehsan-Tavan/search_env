import torch

from src.search.milvus_search import MilvusSearch
from src.configurations import Config


def detect_device() -> str:
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {gpu_name} ({num_gpus} GPU(s) available)")
        return "cuda"
    else:
        print("No GPU detected. Using CPU.")
        return "cpu"


def get_search_client(source: str, config: Config):
    if source == "milvus":
        return MilvusSearch(
            database_name=config.milvus.db_name,
            collection_name=config.milvus.collection_name,
            host=config.milvus.host,
            port=str(config.milvus.port),
            model_name=config.model.name,
            device=detect_device()
        )
    else:
        raise ValueError("Invalid source")
