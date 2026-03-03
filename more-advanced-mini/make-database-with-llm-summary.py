from bs4 import BeautifulSoup

from langchain_ollama import OllamaLLM, OllamaEmbeddings

import chromadb

# read and parse xml file
xmlfile = "/home/Mikyung.Lee/fmscoupler/FMSCoupler/docs/xml/namespaceatm__land__ice__flux__exchange__mod.xml"
with open(xmlfile, "r") as openedfile:
  xmlsoup = BeautifulSoup(openedfile, "lxml-xml")
subroutines = xmlsoup.find_all("memberdef", {"kind": "function"})
variables = xmlsoup.find_all("memberdef", {"kind": "variable"})

# create instance of ollamallm
llm = "qwen2.5"
chatbot = OllamaLLM(model=llm, temperature=0)  

# initialize database, use default embedding function
db_path = "./atm-land-ice-flux-exchange-llm"
collection_name = "atm-land-ice-flux-exchange-llm"
collection = chromadb.PersistentClient(path=db_path).get_or_create_collection(
  name=collection_name,
  metadata = {
    "description": "variables and subroutines from " + collection_name,
    "file": collection_name + ".f90"
  }
)

for variable in variables:
  name = variable.find("name").get_text(strip=True)
  response = chatbot.invoke(f"""
  summarize the variable in this xml information: {variable}.  
  Only include the name of the variable, type of the variable,
  and the briefdescription.
  Do not use special characters or a newline character
  """
  )
  print(response)
  collection.upsert(ids=[name], documents=[response], metadatas=[{"type":"variable"}])
print("added variables to collection")
  
  
for subroutine in subroutines:
  name = subroutine.find("name").get_text(strip=True)
  response = chatbot.invoke(f"""
  Describe the subroutine from this xml information:{subroutine}.  Do not use
  any special characters or a newline character.
  """                            
  )  
  print(f"processed subroutine {name}")
  collection.upsert(ids=[name], documents=[response], metadatas=[{"type":"subroutine"}])
print("added subroutines to colleciton")
                  
