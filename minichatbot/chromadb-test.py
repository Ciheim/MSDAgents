import chromadb
from chromadb import EmbeddingFunction, Documents
from langchain_ollama import OllamaEmbeddings

import numpy as np

llama_embedding = OllamaEmbeddings(model="llama3.2:latest")
client = chromadb.PersistentClient(path="./testdb")

class LlamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, embedding_model):
      self.ef = embedding_model

    def __call__(self, input: Documents):
      return self.ef.embed_documents(input)

def create_database():

    if "animals" in [collection.name for collection in client.list_collections()]: return

    collection = client.create_collection(
      name="animals",
      embedding_function=LlamaEmbeddingFunction(llama_embedding),
      metadata={"subroutine": "animals"}
    )
    collection.add(
      documents=["Cats are okay", "Dogs are awesome"],
      ids=["cats", "dogs"],
      metadatas=[{"type": "cat"}, {"type": "dog"}]
    )

def update_database():

    if "clouds" in [collection.name for collection in client.list_collections()]: return

    collection = client.create_collection(
      name="clouds",
      embedding_function=LlamaEmbeddingFunction(llama_embedding),
      metadata={"subroutine": "clouds"}
    )
    collection.add(
      documents=["Cumulus clouds are puffy", "Stratus clouds are flat"],
      ids=["cumulus", "stratus"],
      metadatas=[{"type": "cumulus"}, {"type": "stratus"}]
    )

def update_cloud_collection():

    collection = client.get_collection(name="clouds", embedding_function=LlamaEmbeddingFunction(llama_embedding))
    collection.update(
    ids=["cirrus"],
    documents=["Cirrus clouds are wispy"],
    metadatas=[{"type": "cirrus"}]
    )

def load_collection_and_query():

    for name in ["animals", "clouds"]:
      collection = client.get_collection(name=name, embedding_function=LlamaEmbeddingFunction(llama_embedding))
      print(collection.query(query_texts=["Are dogs better than cats?"]), "\n\n")

def test_embedding():
    text = ["Cumulus clouds are puffy"]
    embedding_answer = llama_embedding.embed_documents(text)
    embedding_test = client.get_collection(name="clouds").get(ids=["cumulus"], include=["embeddings"])["embeddings"]
    np.testing.assert_array_equal(embedding_answer, embedding_test)

create_database()
update_database()
update_cloud_collection()
load_collection_and_query()
test_embedding()
