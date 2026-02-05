from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate
import chromadb

llama = ChatOllama(model="llama3.2:latest", k=1)
embedding_model = OllamaEmbeddings(model="llama3.2:latest")

use_retriever = False

client = chromadb.PersistentClient("./testdb")
collection_vectorstore = Chroma(
    client=client,
    collection_name="animals",
    embedding_function=embedding_model
)

human_message = HumanMessagePromptTemplate.from_template("{query}. Answer question only using the following content:  {content}")

query = "Are dogs better than cats?"

if use_retriever:
    retriever = collection_vectorstore.as_retriever()
    retrieved = retriever.invoke(query)
else:
    retrieved = collection_vectorstore.similarity_search_with_score(query)

for (document, score) in retrieved:
    print( "SIMILARITY SCORE", score, "DOCUMENT:", document.page_content)

response = llama.invoke([SystemMessage("Be poetic."), human_message.format(query=query, content=retrieved)])
print(response.content)
