from pydantic import BaseModel, Field
import yaml

class MilvusConfig(BaseModel):
    host: str = Field("localhost", description="Milvus host address")
    port: int = Field(19530, description="Milvus server port")
    collection_name: str = Field("my_collection", description="Name of the Milvus collection")
    db_name: str = Field("text_db", description="Milvus database name")
    dim: int = Field(384, description="Embedding dimension")


class ModelConfig(BaseModel):
    name: str = "sentence-transformers/all-MiniLM-L6-v2"

class Config(BaseModel):
    milvus: MilvusConfig = MilvusConfig()
    model: ModelConfig = ModelConfig()


    @staticmethod
    def from_yaml(path: str) -> "Config":
        """Load YAML configuration into a validated Pydantic Config object."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return Config(**raw)