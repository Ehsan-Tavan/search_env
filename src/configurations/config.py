from pydantic import BaseModel, Field, field_validator
import yaml
import os

class MilvusConfig(BaseModel):
    host: str = Field("localhost", description="Milvus host address")
    port: int = Field(19530, description="Milvus server port")
    collection_name: str = Field("my_collection", description="Name of the Milvus collection")
    db_name: str = Field("text_db", description="Milvus database name")
    dim: int = Field(384, description="Embedding dimension")

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError("Port must be between 1 and 65535.")
        return v

    @field_validator("dim")
    @classmethod
    def validate_dim(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Embedding dimension must be positive.")
        return v


class ModelConfig(BaseModel):
    name: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Name or path of the embedding model"
    )


class DataConfig(BaseModel):
    path: str = Field("data.jsonl", description="Path to the JSONL file to ingest")
    chunk_size: int = Field(500, description="Chunk size for document splitting")
    chunk_overlap: int = Field(100, description="Overlap between chunks")

    @field_validator("path")
    @classmethod
    def validate_file_exists(cls, v: str) -> str:
        if not os.path.isfile(v):
            raise FileNotFoundError(f"❌ Data file not found: {v}")
        return v

    @field_validator("chunk_size", "chunk_overlap")
    @classmethod
    def validate_positive(cls, v: int, info) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be greater than 0.")
        return v


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