from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain_core.vectorstores import InMemoryVectorStore

"""
#to run locally 
podman pull ollama:latest (can skip this step on the amdbox, it's already pulled?)
./ollama-run
./ollama pull llama3.2:latest

#set up environment 
pip install . (or wherever the pyproject.toml file is)

#run script
python run.py
"""

llm_model = "llama3.2:latest"

#load model, create embedding_model
llama3 = ChatOllama(model=llm_model)
embedding_model = OllamaEmbeddings(model=llm_model)

#chat with llama3
response = llama3.invoke("Hello!")
print("default")
print(response.content)
print("*****")

#chat with system and human messages
messages = [
    SystemMessage(content="Be sarcastic and always say thank you at the end of your reply."),
    HumanMessage(content="Hello!"),
]
response = llama3.invoke(messages)
print("using prompts, be sarcastic")
print(response.content)
print("*****")

#use prompts instead
system_prompt = SystemMessagePromptTemplate.from_template("Be {attitude} and always say thank you at the end of your reply.")
human_prompt = HumanMessagePromptTemplate.from_template("{query}.  Use this as {content}")
messages = [
    system_prompt.format(attitude="sad"),
    human_prompt.format(query="Hello!", content=""),
]
response = llama3.invoke(messages)
print("using templates to generate prompts, be sad")
print(response.content)
print("*****")

document = """
    FMSCoupler provides the capability to couple component models (atmosphere, land, sea ice, and ocean)
    on different logically rectangular grids.  Sally sells seashells by the seashore.
"""

#create vectorstore to store embedded data and perform similarity search
print("retrieved from the vectorstore")
vectorstore = InMemoryVectorStore.from_texts([document], embedding=embedding_model)
retriever = vectorstore.as_retriever()
responses = retriever.invoke("What is FMSCoupler?")
for response in responses: print(response.page_content)
print("*****")

print("content provided chatbot with helpful attitude")
messages = [
  system_prompt.format(attitude="helpful"),
  human_prompt.format(query="What is FMSCoupler?", content=responses),
]
response = llama3.invoke(messages)
print(response.content)
print("*****")


#explore embedding
#hello_world = embedding_model.embed_query("Hello world")
#print(hello_world, "\n", "len(hello_world):", len(hello_world), "\n", end=printending)
#import spacy
#import spacy.cli
#from langchain_core.documents import Document
#spacy.cli.download("en_core_web_sm")
#    vectorstore = InMemoryVectorStore(embedding=embedding_model)
#    nlp = spacy.load("en_core_web_sm")
#    splitted_text = [sent.text for sent in nlp(document).sents]
#    vectorstore.add_documents([Document(page_content=text) for text in splitted_text])
#    retriever = vectorstore.as_retriever()
#    responses = retriever.invoke("What is FMSCoupler?")
#    print(responses, end=printending)

#    vectorstore = InMemoryVectorStore(embedding=embedding_model)
#    nlp = spacy.load("en_core_web_sm")
#    splitted_text = [sent.text for sent in nlp(document).sents]
#    vectorstore.add_documents([Document(page_content=text) for text in splitted_text])
#    result = vectorstore.similarity_search("What is FMSCoupler?")
#    print(result, end=printending)







