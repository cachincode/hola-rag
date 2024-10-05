import os

from llama_index.core import (
SimpleDirectoryReader,
VectorStoreIndex,
StorageContext,
load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

data_path = "/home/chachin/Documents/chachin_doc/hongos_medicinales/papers"
embedding_path = "./model"
index_store_path = "./storage"

documents = SimpleDirectoryReader(data_path).load_data(show_progress=True)

embed_model = HuggingFaceEmbedding(cache_folder=embedding_path)


if os.path.isdir(index_store_path):
    print("loading indexes from storage...")
    storage_context = StorageContext.from_defaults(persist_dir=index_store_path)

    vector_index = load_index_from_storage(
        storage_context=storage_context,
        embed_model=embed_model
    )

else:
    print("creating indexes...")
    vector_index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
        embed_model=embed_model
    )

    vector_index.storage_context.persist(persist_dir=index_store_path)
