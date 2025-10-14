import argparse
from tqdm import tqdm
from itertools import islice

from src.configurations import Config
from src.vector_db_handler import MilvusHandler
from src.vector_db_handler import Embedder
from src.data_loader import JSONLLoader, Chunker
from src.data_loader import clean_text


def batched(iterable, batch_size):
    """Yield successive batches from iterable."""
    it = iter(iterable)
    while batch := list(islice(it, batch_size)):
        yield batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search Environment"
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        type=str,
        help="Config file path (default: None)"
    )

    args = parser.parse_args()

    # Validate required arguments
    if args.config is None:
        raise ValueError("The config argument should be set!")

    config = Config.from_yaml(args.config)

    milvus_obj = MilvusHandler(config=config.milvus)
    embedder = Embedder(config.model.name)
    chunker = Chunker(chunk_size=config.data.chunk_size,
                      chunk_overlap=config.data.chunk_overlap,
                      model_name=config.model.name)

    texts = JSONLLoader.load_texts(config.data.path, text_field="text")

    # Batch processing to avoid memory explosion
    batch_size = 256
    for text_batch in tqdm(batched(texts, batch_size), desc="Processing text batches"):
        # Clean
        clean_texts = [clean_text(t) for t in text_batch]

        # Chunk
        chunks = chunker.chunk_text(clean_texts)

        # Embed
        embeddings = embedder.encode_texts(chunks)

        # Insert into Milvus
        milvus_obj.insert_data(embeddings, chunks)


    # clean_texts = [clean_text(text) for text in tqdm(texts, desc="Cleaning texts", ncols=80)]
    #
    # chunks = chunker.chunk_text(texts=clean_texts)
    #
    # embeddings = embedder.encode_texts(chunks)
    #
    # milvus_obj.insert_data(embeddings, chunks)
