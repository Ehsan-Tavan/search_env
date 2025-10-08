import unittest
from unittest.mock import patch, MagicMock
from src.vector_db_handler.milvus_client import MilvusHandler
from src.configurations import MilvusConfig


class TestMilvusClient(unittest.TestCase):
    def setUp(self):
        self.config = MilvusConfig(
            host="localhost",
            port=19530,
            db_name="test_db",
            collection_name="test_collection",
            dim=3
        )
        self.handler = MilvusHandler(self.config)

    # ---- Connection ----
    @patch("src.vector_db_handler.milvus_client.connections.connect")
    def test_connect_calls_pymilvus_connect(self, mock_connect):
        self.handler._connect()
        mock_connect.assert_called_once_with(
            alias="default",
            host=self.config.host,
            port=str(self.config.port)
        )

    # ---- Database ----
    @patch("src.vector_db_handler.milvus_client.db")
    def test_create_database_if_not_exists_creates(self, mock_db):
        mock_db.list_database.return_value = []
        self.handler._create_database_if_not_exists("test_db")
        mock_db.create_database.assert_called_once_with("test_db")
        mock_db.using_database.assert_called_once_with("test_db")

    @patch("src.vector_db_handler.milvus_client.db")
    def test_create_database_if_already_exists(self, mock_db):
        mock_db.list_database.return_value = ["test_db"]
        self.handler._create_database_if_not_exists("test_db")
        mock_db.create_database.assert_not_called()
        mock_db.using_database.assert_called_once_with("test_db")

    # ---- Collection ----
    @patch("src.vector_db_handler.milvus_client.utility.has_collection", return_value=True)
    @patch("src.vector_db_handler.milvus_client.Collection")
    def test_create_collection_if_exists_loads_existing(self, mock_collection, mock_has):
        self.handler._create_collection_if_not_exists("test_collection", 3)
        mock_collection.assert_called_once_with("test_collection")
        self.assertEqual(self.handler.collection, mock_collection.return_value)
        mock_collection.return_value.load.assert_called_once_with(replica_number=1)

    @patch("src.vector_db_handler.milvus_client.utility.has_collection", return_value=False)
    @patch("src.vector_db_handler.milvus_client.Collection")
    @patch("src.vector_db_handler.milvus_client.CollectionSchema")
    @patch("src.vector_db_handler.milvus_client.FieldSchema")
    def test_create_collection_if_not_exists_creates_new(self, mock_field, mock_schema, mock_collection, mock_has):
        self.handler._create_collection_if_not_exists("new_collection", 3)
        mock_collection.assert_called_once()
        mock_collection.return_value.create_index.assert_called_once()
        mock_collection.return_value.load.assert_called_once_with(replica_number=1)

    # ---- Configuring DB ----
    @patch.object(MilvusHandler, "_create_collection_if_not_exists")
    @patch.object(MilvusHandler, "_create_database_if_not_exists")
    @patch.object(MilvusHandler, "_connect")
    def test_configuring_db_calls_all(self, mock_connect, mock_create_db, mock_create_col):
        self.handler._configuring_db()
        mock_connect.assert_called_once()
        mock_create_db.assert_called_once_with(self.config.db_name)
        mock_create_col.assert_called_once_with(
            name=self.config.collection_name,
            dim=self.config.dim
        )

    # ---- Insert Data ----
    def test_insert_data_raises_if_length_mismatch(self):
        self.handler.collection = MagicMock()
        with self.assertRaises(ValueError):
            self.handler.insert_data([[0.1, 0.2]], ["text1", "text2"])

    def test_insert_data_calls_insert_and_flush(self):
        mock_collection = MagicMock()
        mock_collection.name = "mock_collection"
        self.handler.collection = mock_collection

        embeddings = [[0.1, 0.2, 0.3]]
        texts = ["text1"]
        self.handler.insert_data(embeddings, texts)

        mock_collection.insert.assert_called_once_with([embeddings, texts])
        mock_collection.flush.assert_called_once()

    @patch.object(MilvusHandler, "_configuring_db")
    def test_insert_data_auto_configures(self, mock_configure_db):
        # collection is None, should trigger _configuring_db
        embeddings = [[0.1, 0.2, 0.3]]
        texts = ["text1"]

        # after configuration, simulate that collection exists
        mock_collection = MagicMock()
        self.handler.collection = mock_collection

        # temporarily force _configuring_db to assign collection
        def fake_config():
            self.handler.collection = mock_collection
        mock_configure_db.side_effect = fake_config

        self.handler.collection = None
        self.handler.insert_data(embeddings, texts)

        mock_configure_db.assert_called_once()
        mock_collection.insert.assert_called_once()
        mock_collection.flush.assert_called_once()


if __name__ == "__main__":
    unittest.main()
