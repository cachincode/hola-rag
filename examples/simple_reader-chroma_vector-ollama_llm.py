import os

import chromadb
from llama_index.core import (
SimpleDirectoryReader,
VectorStoreIndex,
StorageContext,
load_index_from_storage
)
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts import PromptTemplate
from transformers import BitsAndBytesConfig
import torch

data_path = "./data"
embedding_path = "./model"
persist_storage_path = "./storage"
vector_store_path = "./chroma_db"
storage_type = "db" # "db" || "persist"


documents = None
if storage_type == "persist" or not os.path.isdir(vector_store_path):
    documents = SimpleDirectoryReader(data_path).load_data(show_progress=True)

embed_model = HuggingFaceEmbedding(cache_folder=embedding_path)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
llm = HuggingFaceLLM(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
    query_wrapper_prompt=PromptTemplate("<s>[INST] {query_str} [/INST] </s>\n"),
    model_kwargs={"quantization_config": quantization_config},
)

index = None
if storage_type == "persist":

    if os.path.isdir(persist_storage_path):
        print("loading indexes from persiste disk...")

        storage_context = StorageContext.from_defaults(persist_dir=persist_storage_path)
        index = load_index_from_storage(
            storage_context=storage_context,
            embed_model=embed_model
        )
    else:
        print("creating indexes and persisting them...")

        index = VectorStoreIndex.from_documents(
            documents,
            show_progress=True,
            embed_model=embed_model
        )
        index.storage_context.persist(persist_dir=persist_storage_path)

elif storage_type == "db":

    if os.path.isdir(vector_store_path):
        print("loading vector store from database")

        db = chromadb.PersistentClient(path=vector_store_path)
        chroma_collection = db.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
            storage_context=storage_context
        )

    else:
        print("creating vector store and indexes...")

        db = chromadb.PersistentClient(path=vector_store_path)
        chroma_collection = db.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents=documents,
            embed_model=embed_model,
            storage_context=storage_context,
        )

query_engine = index.as_query_engine(llm=llm, streaming=True)

response = query_engine.query()

response.print_response_stream()
