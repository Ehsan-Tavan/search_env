# 🔍 search_env — A Search Engine API for LLMs

`earch_env` is a **search and retrieval API** that creates a **external environment** for **Large Language Models (LLMs)** to interact with **vector-based search systems** during reasoning.

It enables the development of **SearchLLMs** — models that can:  
- 🔍 Query external information sources,  
- 🧩 Retrieve and reason over factual evidence,  
- 🧠 Generate grounded, explainable answers.  

By combining **LLMs**, **Milvus vector search**, and **retrieval-augmented reasoning**, `search_env` bridges the gap between static knowledge and dynamic information retrieval.


## ✨ Key Features

✅ **Multilingual Embeddings —** Uses the multilingual-e5-small model to encode text into 384-dimensional embeddings, enabling efficient multilingual retrieval (e.g., Persian, English).  
✅ **Token-Based Chunking —** Documents are split into manageable segments using a token chunker to preserve semantic coherence and improve retrieval accuracy.  
✅ **Modular Architecture —** Plug-and-play search strategies (SimpleSearch, HashSearch, or custom).  
✅ **API-Driven —** Exposes clean REST endpoints for search, index, and health-check operations.  
✅ **LLM Integration Ready —** Built for reasoning loops that involve querying and retrieving.  
✅ **Configurable —** All parameters managed through YAML configs and environment variables.  
✅ **Docker Support —** One-line setup for reproducible environments.  
✅ **Scalable Design —** Easily extend or integrate new backends (e.g., vector DBs, hybrid retrieval).

## 🧩 Document Chunking and Embedding
To handle large documents efficiently, `search_env` **chunks text by tokens** before embedding. This ensures that each chunk fits within the model’s context window and retains coherent meaning.

Below is the **token-based** chunker used in the pipeline:

```python
def chunk_text(self, texts):
    all_chunks = []
    for doc in texts:
        # Encode into tokens
        tokens = self.tokenizer.encode(doc, add_special_tokens=False)

        # Chunk by token length
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            all_chunks.append(chunk_text.strip())

    return all_chunks
```

Each document is split into overlapping token chunks (`chunk_size` and `chunk_overlap` configurable via YAML).  
Then, each chunk is embedded using **multilingual-e5-small**, a lightweight and high-quality multilingual model.

This combination improves retrieval precision and efficiency across languages and long documents.



## 🚀 Quick Start
Clone the repository and launch the API to get up and running quickly:

```bash
git clone https://gitlab.partdp.ir/naturallanguageprocessing/lm-training/search_env/
cd search_env
pip install -r requirements.txt
```

### 1️⃣ Start Milvus and Dependencies via Docker Compose  
After cloning the repo, go to the `docker_compose` directory and run:

```bash
cd docker_compose
docker compose up -d
```

This command will spin up `Milvus`, `etcd`, and `MinIO` containers required by the system.  
Once the services are running, you can proceed to populate the Milvus database with your data.

### 2️⃣ Add Data to Milvus
To index your data into Milvus, run the following command:
```bash
python save_data_to_db.py --config configs/config.yaml
```

This script loads the dataset specified in your config file, chunks it, embeds it, and stores the resulting vectors into Milvus for retrieval.


Then, you can run search_env either via Docker or locally:

[//]: # (### 1. Docker:)

[//]: # (Build and run the containerized API:)

[//]: # (```bash)

[//]: # (docker build -t search_env .)

[//]: # (docker run -p 8000:8000 search_env)

[//]: # (```)

[//]: # ()
[//]: # (The API will be available at http://localhost:8000.)


### 3️⃣ Run the Search API

Once your data is indexed, start the API server:

```bash
python -m src.app.main --config configs/config.yaml
```

This also starts the server on port `5250` by default.


⚙️ Configuration Explained (`configs/config.yaml`)

Your configuration defines **three main components:** the vector database (Milvus), the embedding model, and the input data used for indexing and search.

### 🧩 Example

```yaml
milvus:
  host: 127.0.0.1
  port: 19530
  collection_name: persian_wikipedia
  db_name: Search_Env

model:
  name: model_path
  dim: 384

data:
  path: data_path
  chunk_size: 256
  chunk_overlap: 32
```

#### 🔍 Breakdown

**1. Milvus Configuration**

| Key               | Description                                                                               |
| ----------------- | ----------------------------------------------------------------------------------------- |
| `host`            | IP address of the Milvus vector database (default: `127.0.0.1`)                           |
| `port`            | Milvus service port (default: `19530`)                                                    |
| `collection_name` | Name of the Milvus collection that stores your text embeddings (e.g. `persian_wikipedia`) |
| `db_name`         | Logical database name inside Milvus (e.g. `Search_Env`)                                   |


🧠 *Milvus is a high-performance vector database optimized for similarity search. `search_env` uses it to store and retrieve embeddings efficiently.*

---

**2. Model Configuration**

| Key    | Description                                                              |
| ------ | ------------------------------------------------------------------------ |
| `name` | Path to the embedding model (e.g., HuggingFace model or local directory) |
| `dim`  | Dimensionality of the embedding vectors produced by the model            |

💡 *In your setup, **multilingual-e5-small** (dimension 384) is used for embedding Persian Wikipedia text, making the system multilingual and lightweight.*

---

**3. Data Configuration**

| Key             | Description                                                            |
| --------------- | ---------------------------------------------------------------------- |
| `path`          | Path to the input dataset (JSONL format) containing raw text documents |
| `chunk_size`    | Number of tokens or characters per text chunk during indexing          |
| `chunk_overlap` | Number of overlapping tokens/characters between consecutive chunks     |


📚 *Chunking allows better retrieval granularity — long documents are broken into smaller pieces so that the LLM retrieves only relevant portions of text.*


## 🧠 Example Workflow

- 1. **Load data** from `data.path`

- 2. **Chunk text** using `chunk_size` and `chunk_overlap`

- 3. **Encode each chunk** into a 384-dimensional vector using `model.name`

- 4. **Store vectors** in **Milvus** under the specified `collection_name`

- 5. **Query** Milvus for the top-k similar embeddings during search

## 🔍 Example API Usage

**Search**

```bash
curl -X POST http://localhost:5250/search \
     -H "Content-Type: application/json" \
     -d '{"query": "پیشرفت‌های اخیر در هوش مصنوعی"}'
```

**Example Response**

```json
[
    {
        "id": 461347537240209218,
        "score": 0.873076856136322,
        "text": "text",
        "meta": {}
    },
    {
        "id": 461347537240210140,
        "score": 0.860804557800293,
        "text": "text",
        "meta": {}
    },
    {
        "id": 461347537240209422,
        "score": 0.8587036728858948,
        "text": "text",
        "meta": {}
    },
    {
        "id": 461347537240209310,
        "score": 0.8570656776428223,
        "text": "text",
        "meta": {}
    },
    {
        "id": 461347537240210117,
        "score": 0.855384349822998,
        "text": "text",
        "meta": {}
    }
]
```

## 🧩 Extend and Customize

Want to experiment with different retrieval techniques?

### 1 .Add a new search method:

```python
from .base import SearchClient, SearchResult

class BM25Search(SearchClient):
    def __init__(self, corpus):
        self.corpus = corpus
        # Initialize BM25 index here

    def search(self, query: str, top_k: int = 5):
        # Replace with your logic
        results = self.corpus.get_top_k(query, k=top_k)
        return [
            SearchResult(id=i, score=s, text=t)
            for i, (t, s) in enumerate(results)
        ]

    def warmup(self):
        print("BM25 index loaded.")
```

Then update your `config.yaml` or runtime argument to use your new method:

```yaml
search_method: "BM25Search"
```

and run:

```bash
python -m src.app.main --config configs/config.yaml
```

## 🧪 Running Tests

The `tests` package contains unittests for all major modules in `search_env`, including the Milvus client, configuration loader, and search components.

You can run all tests from project root:

```bash
python -m unittest -v
```

🧠 *This command automatically discovers and runs all files matching the pattern test*.py inside `search_env` project (e.g., src/tests/test_vector_db_handler.py).