from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

#https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
#https://docs.trychroma.com/guides/build/intro-to-retrieval

chroma_client = chromadb.Client()

embedding_model = OllamaEmbeddings(model="llama3.2:latest")

keyword = "FMSCoupler"
question = f"What is {keyword}?"
documents = [
    Document("FMSCoupler provides the capability to couple component models (atmosphere, land, sea ice, and ocean).", metadata={"source": keyword}, id=0),
    Document("Sally sells seashells by the seashore.", metadata={"source": "tongue twister"}, id=1),
    Document("The quick brown fox jumps over the lazy dog.", metadata={"source": "pangram"}, id=2),
    Document("LangChain is a framework for building applications with LLMs.", metadata={"source": "langchain"}, id=3),
    Document("Mr. Darcy makes ten thousand pounds a year", metadata={"source": "pride and prejudice"}, id=4),
]

vector_store = Chroma(
    embedding_function=embedding_model,
    collection_name="fmscoupler_collection"
)
vector_store.add_documents(documents)

#similarity search with score
print(question)
responses_and_scores = vector_store.similarity_search_with_score(question, k=len(documents))
for (response, score) in responses_and_scores: print(f"Score: {score}, {response.page_content}")


#testing normalization
def test_nomalization(documents: list[Document]):
    import torch
    for document in documents:
        testing = embedding_model.embed_query(document.page_content)
        norm = torch.linalg.norm(torch.tensor(testing))
        print(f"Norm of the embedded vector for document {document.id} '{document.metadata['source']}': {norm}")