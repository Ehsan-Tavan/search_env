from pydantic import BaseModel, Field
import yaml

class MilvusConfig(BaseModel):
    host: str = Field("localhost", description="Milvus host address")
    port: int = Field(19530, description="Milvus server port")
    collection_name: str = Field("my_collection", description="Name of the Milvus collection")
    db_name: str = Field("text_db", description="Milvus database name")
    dim: int = Field(384, description="Embedding dimension")


class ModelConfig(BaseModel):
    name: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Name or path of the embedding model"
    )


class DataConfig(BaseModel):
    path: str = Field("data.jsonl", description="Path to the JSONL file to ingest")
    chunk_size: int = Field(500, description="Chunk size for document splitting")
    chunk_overlap: int = Field(100, description="Overlap between chunks")


class Config(BaseModel):
    milvus: MilvusConfig = MilvusConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()


    @staticmethod
    def from_yaml(path: str) -> "Config":
        """Load YAML configuration into a validated Pydantic Config object."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return Config(**raw)