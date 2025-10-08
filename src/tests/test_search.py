import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from src.search.milvus_search import MilvusSearch, SearchResult
from src.search.search_tool import detect_device, searching_tool


class TestMilvusSearch(unittest.TestCase):
    """Unit tests for MilvusSearch"""

    def setUp(self):
        # Patch SentenceTransformer and Collection before each test
        patcher_model = patch("src.search.milvus_search.SentenceTransformer")
        patcher_collection = patch("src.search.milvus_search.Collection")

        self.addCleanup(patcher_model.stop)
        self.addCleanup(patcher_collection.stop)

        self.mock_model_class = patcher_model.start()
        self.mock_collection_class = patcher_collection.start()

        # Mock SentenceTransformer instance
        self.mock_model_instance = self.mock_model_class.return_value
        self.mock_model_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

        # Mock Collection instance
        self.mock_collection_instance = MagicMock()
        self.mock_collection_class.return_value = self.mock_collection_instance

    @patch("src.search.milvus_search.connections.connect")
    @patch("src.search.milvus_search.db.using_database")
    def test_init(self, mock_db, mock_connect):
        """Test MilvusSearch initialization"""
        client = MilvusSearch(
            database_name="test_db",
            collection_name="test_collection",
            host="127.0.0.1",
            port="19530",
            model_name="test-model",
            device="cpu",
        )

        mock_connect.assert_called_once_with(host="127.0.0.1", port="19530")
        mock_db.assert_called_once_with("test_db")
        self.assertEqual(client.collection_name, "test_collection")
        self.assertEqual(client.model, self.mock_model_instance)

    @patch("src.search.milvus_search.connections.connect")
    @patch("src.search.milvus_search.db.using_database")
    def test_search(self, mock_db, mock_connect):
        """Test the search() method returns correct results"""
        hit = MagicMock()
        hit.id = "10"
        hit.distance = 0.88
        hit.entity.get.return_value = "Mocked document"
        self.mock_collection_instance.search.return_value = [[hit]]

        client = MilvusSearch()
        results = client.search("query", top_k=3)

        self.mock_model_instance.encode.assert_called_once_with(["query"], convert_to_numpy=True)
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertIsInstance(res, SearchResult)
        self.assertEqual(res.id, 10)
        self.assertEqual(res.text, "Mocked document")

    @patch("src.search.milvus_search.connections.connect")
    @patch("src.search.milvus_search.db.using_database")
    def test_warmup(self, mock_db, mock_connect):
        """Test the warmup() method loads the collection"""
        client = MilvusSearch()
        client.warmup()
        self.mock_collection_instance.load.assert_called_once()


class TestSearchTool(unittest.TestCase):
    """Unit tests for search_tool.py"""

    @patch("torch.cuda.is_available", return_value=False)
    def test_detect_device_cpu(self, mock_is_available):
        self.assertEqual(detect_device(), "cpu")

    @patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090")
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.is_available", return_value=True)
    def test_detect_device_gpu(self, mock_is_available, mock_device_count, mock_get_device_name):
        self.assertEqual(detect_device(), "cuda")

    @patch("src.search.search_tool.MilvusSearch")
    @patch("src.search.search_tool.detect_device", return_value="cpu")
    def test_searching_tool_milvus(self, mock_detect_device, mock_milvus):
        """Should create MilvusSearch client and perform search correctly"""
        mock_client = MagicMock()
        mock_client.search.return_value = [
            {"id": 1, "score": 0.9, "text": "Example text"}
        ]
        mock_milvus.return_value = mock_client

        # Mock config structure
        class MockMilvusConfig:
            db_name = "test_db"
            collection_name = "test_collection"
            host = "localhost"
            port = 19530

        class MockConfig:
            milvus = MockMilvusConfig()
            model = type("Model", (), {"name": "test-model"})

        results = searching_tool("query", source="milvus", top_k=3, config=MockConfig())

        mock_milvus.assert_called_once()
        mock_client.search.assert_called_once_with("query", top_k=3)
        self.assertIsInstance(results, list)
        self.assertIn("text", results[0])
        self.assertEqual(results[0]["text"], "Example text")

    def test_searching_tool_invalid_source(self):
        with self.assertRaises(ValueError):
            searching_tool("query", source="invalid", config=None)


if __name__ == "__main__":
    unittest.main()
