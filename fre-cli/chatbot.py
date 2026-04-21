import chromadb

#import langchain
from langchain_ollama import ChatOllama
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate


llm = "llama3.2"
collection_name = "fremake"
db_path = "./fremake_database"


system_message = SystemMessagePromptTemplate.from_template(
    """
    Only answer questions about fre make.
    If the user asks questions not related to fre make, 
    Say you can only answer questions about fre make.
    TODO:  add instructions on how to provide usage examples
    """
)

human_message = HumanMessagePromptTemplate.from_template(
    """
    {query}.  
    Only use this content to answer the question: {content}.
    Do not add any external knowledge.
    """
)


#load database
client = chromadb.PersistentClient(db_path)
collections = client.list_collections()

#create chatbot 
chatbot = ChatOllama(model=llm, temperature=0.0)

query = "Say hello"
print(f"{chatbot.invoke(query).content}\n>", end=" ")

messages = [system_message.format()]

while True:
    query = input()
    if "goodbye" in query.lower(): exit        

    content = collections[0].query(query_texts=[query])["documents"]
    messages.append(human_message.format(query=query, content=content))
    answer = chatbot.invoke(messages)
    messages.append(answer)
    print(answer.content + "\n >", end=" ")

