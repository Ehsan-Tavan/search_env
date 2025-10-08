import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.search.milvus_search import MilvusSearch, SearchResult
from src.search.search_tool import detect_device, searching_tool


class TestMilvusSearch:
    """Unit tests for MilvusSearch"""

    @pytest.fixture
    def mock_sentence_transformer(self):
        with patch("src.search.milvus_search.SentenceTransformer") as mock_model:
            instance = mock_model.return_value
            instance.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
            yield instance

    @pytest.fixture
    def mock_collection(self):
        with patch("src.search.milvus_search.Collection") as mock_col:
            collection_instance = MagicMock()
            mock_col.return_value = collection_instance
            yield collection_instance

    @patch("src.search.milvus_search.connections.connect")
    @patch("src.search.milvus_search.db.using_database")
    def test_init(
            self, mock_db, mock_connect, mock_collection, mock_sentence_transformer
    ):
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
        assert client.collection_name == "test_collection"
        assert client.model == mock_sentence_transformer

    @patch("src.search.milvus_search.connections.connect")
    @patch("src.search.milvus_search.db.using_database")
    def test_search(self, mock_db, mock_connect, mock_collection, mock_sentence_transformer):
        """Test the search() method returns correct results"""
        hit = MagicMock()
        hit.id = "10"
        hit.distance = 0.88
        hit.entity.get.return_value = "This is a test document"
        mock_collection.search.return_value = [[hit]]

        client = MilvusSearch()
        results = client.search("sample query", top_k=3)

        # Ensure encode was called correctly
        mock_sentence_transformer.encode.assert_called_once_with(
            ["sample query"], convert_to_numpy=True
        )

        assert len(results) == 1
        res = results[0]
        assert isinstance(res, SearchResult)
        assert res.id == 10
        assert pytest.approx(res.score, 0.88)
        assert res.text == "This is a test document"

    @patch("src.search.milvus_search.connections.connect")
    @patch("src.search.milvus_search.db.using_database")
    def test_warmup(self, mock_db, mock_connect, mock_collection, mock_sentence_transformer):
        """Test the warmup() method loads the collection"""
        client = MilvusSearch()
        client.warmup()
        mock_collection.load.assert_called_once()


class TestSearchTool:
    """Unit tests for search_tool.py"""

    def test_detect_device_cpu(self, monkeypatch):
        """Should return 'cpu' when CUDA is unavailable"""
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        assert detect_device() == "cpu"

    def test_detect_device_gpu(self, monkeypatch):
        """Should return 'cuda' when GPU is detected"""
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
        monkeypatch.setattr("torch.cuda.get_device_name", lambda _: "NVIDIA RTX 4090")
        assert detect_device() == "cuda"

    @patch("src.search.search_tool.MilvusSearch")
    @patch("src.search.search_tool.detect_device", return_value="cpu")
    def test_searching_tool_milvus(self, mock_detect_device, mock_milvus):
        """Should create MilvusSearch client and perform search correctly"""
        mock_client = MagicMock()
        mock_client.search.return_value = [
            MagicMock(id=1, score=0.9, text="Example text")
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
        assert isinstance(results, list)
        assert isinstance(results[0], dict)
        assert "text" in results[0]

    def test_searching_tool_invalid_source(self):
        """Should raise ValueError for unsupported source"""
        with pytest.raises(ValueError, match="Invalid source"):
            searching_tool("test", source="invalid", config=None)
