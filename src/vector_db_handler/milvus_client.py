from pymilvus import (connections, db, FieldSchema, CollectionSchema, DataType,
                      Collection, utility)

from src.configurations import MilvusConfig


class MilvusHandler:
    def __init__(self, config: MilvusConfig):
        self.config = config

        self.collection = None

    def _connect(self):
        connections.connect(
            alias="default",
            host=self.config.host,
            port=str(self.config.port)
        )
        print(f"Connected to Milvus at {self.config.host}:{self.config.port}")

    def _create_database_if_not_exists(self, name: str):
        """Create Milvus database if it doesn't already exist."""
        try:
            if name not in db.list_database():
                db.create_database(name)
                print(f"Created new database '{name}'")
            else:
                print(f"Database '{name}' already exists.")
            db.using_database(name)
        except AttributeError:
            print("Database management not supported in this Milvus version. Using default DB.")

    def _create_collection_if_not_exists(self, name, dim):
        """Create or load the collection and store it in self.collection."""
        if utility.has_collection(name):
            print(f"Collection '{name}' already exists.")
            self.collection = Collection(name)

        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),
            ]

            schema = CollectionSchema(fields, description="Text embeddings collection")
            self.collection = Collection(name, schema)
            print(f"Created new collection '{name}' with dim={dim}")

            # Create an index for efficient vector search
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 128},
            }
            self.collection.create_index("embedding", index_params)
            print("Created index on 'embedding' field (IVF_FLAT, COSINE)")

        self.collection.load(replica_number=1)
        print(f"Loaded collection '{name}' successfully.")

    def _configuring_db(self):
        """Ensure connection, DB, and collection are ready before inserting."""
        self._connect()
        self._create_database_if_not_exists(self.config.db_name)
        self._create_collection_if_not_exists(
            name=self.config.collection_name,
            dim=self.config.dim
        )

    def insert_data(self, embeddings, texts):
        """Insert text embeddings into Milvus."""
        if self.collection is None:
            self._configuring_db()

        if len(embeddings) != len(texts):
            raise ValueError("Number of embeddings and texts must match.")

        # Insert data
        data = [embeddings, texts]
        self.collection.insert(data)
        self.collection.flush()
        print(f"Inserted {len(texts)} records into '{self.collection.name}'.")


