from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate
import chromadb


llama = ChatOllama(model="llama3.2:latest", k=1)
embedding_model = OllamaEmbeddings(model="llama3.2:latest")

client = chromadb.PersistentClient("./testdb")
collection_vectorstore = Chroma(
  client=client,
  collection_name="animals",
  embedding_function=embedding_model
)


retriever = collection_vectorstore.as_retriever()
print(retriever.invoke("What is a dog?"))

SystemMessage("Be cheerful!"),
human_message = HumanMessagePromptTemplate.from_template("{query}, use only {content} to answer question.")

query = "Are dogs better than cats?"
retrieved = retriever.invoke(query)
response = llama.invoke([SystemMessage("Be cheerful!"), human_message.format(query=query, content=retrieved)])
print(response.content)
