import torch
from transformers import BitsAndBytesConfig
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.core import VectorStoreIndex

class RAG:
    def __init__(self) -> None:
        self.urls = []
        self.llm = None
        self.embed_model = None
        self.query_engine = None

    def load_models(self):
        print("Cargando modelos...")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.llm = HuggingFaceLLM(
            model_name="mistralai/Mistral-7B-Instruct-v0.1",
            tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
            query_wrapper_prompt=PromptTemplate("<s>[INST] {query_str} [/INST] </s>\n"),
            model_kwargs={"quantization_config": quantization_config},
        )

        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_folder="./model")

        print("Modelos cargados. Agrega documentos para indexar...")

    def add_url(self, url):
        if url not in self.urls:
            self.urls.append(url)
            print(f"URL añadida: {url}")
            self.update_index()
        else:
            print("URL ya se encuentra indexada.")

    def update_index(self):
        if len(self.urls) > 0:
            documents = BeautifulSoupWebReader().load_data(self.urls)
            vector_index = VectorStoreIndex.from_documents(
                documents=documents,
                embed_model=self.embed_model
            )
            self.query_engine = vector_index.as_query_engine(llm=self.llm, streaming=True)

    def ask_question(self, question):
        if self.query_engine is None:
            print("Motor de queries no ha sido inicializado. Ya tienes documentos indexados?...")
            return
        response = self.query_engine.query(question)
        response.print_response_stream()

def main():
    rag = RAG()
    rag.load_models()

    while True:
        user_input = input("\nQué quieres hacer? (pregunta/añadir/salir): ").lower()

        if user_input == "pregunta":
            question = input("Ingresa tu pregunta: ")
            rag.ask_question(question)
        elif user_input == "añadir":
            url = input("Ingresa la URL: ")
            rag.add_url(url)
        elif user_input == "salir":
            print("Saliendo del programa. Adiós!")
            break
        else:
            print(f"{user_input} es una opción inválida ('pregunta', 'añadir', o 'salir').")

if __name__ == "__main__":
    main()
