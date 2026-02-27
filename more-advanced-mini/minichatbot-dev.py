from bs4 import BeautifulSoup

import chromadb

from langchain_ollama import ChatOllama
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.messages import SystemMessage

llm = "llama3.2"
collection_name = "atm-land-ice-flux-exchange"
db_path = "./atm-land-ice-flux-exchange"

chatbot = ChatOllama(model=llm, microstat_tau=2.0)

client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(collection_name)

system_message = SystemMessagePromptTemplate.from_template(
  """
  You are a chatbot who only answer questions related
  to the FMSCoupler program.  FMSCoupler program is a Fortran 
  program developed at GFDL for climate modeling.
  """
)

human_message = HumanMessagePromptTemplate.from_template(
  """
  {query}.  Only use this content to answer the question:  {content}.
  Do not add any external knowledge.
  """
)

query = "Introduce yourself.  Ask how may I assist you?"
intro = chatbot.invoke([
  system_message.format(),
  human_message.format(query=query, content="")
])
print(intro.content + "\n>", end=" ")

while "goodbye" not in query.lower():

  query = input()
  retrieved = collection.query(query_texts=[query])["documents"]
  print(retrieved)
  answer = chatbot.invoke([
    system_message.format(),
    human_message.format(query=query, content=retrieved)
  ])
  print(answer.content)
  print(">", end=" ")
               





    










    
    
       


