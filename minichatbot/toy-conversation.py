from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain_core.vectorstores import InMemoryVectorStore

"""
#to run locally
podman pull ollama:rocm (can skip this step on the amdbox, it's already pulled?)
./ollama-run
./ollama pull llama3.2:latest

#set up environment
pip install . (or wherever the pyproject.toml file is)

#run script
python run.py
"""

document = """
    FMSCoupler provides the capability to couple component models (atmosphere, land, sea ice, and ocean)
    on different logically rectangular grids.  Sally sells seashells by the seashore.
"""

llm_model = "llama3.2:latest"

#load model, create embedding_model
llama3 = ChatOllama(model=llm_model)
embedding_model = OllamaEmbeddings(model=llm_model)

#retriever
vectorstore = InMemoryVectorStore.from_texts([document], embedding=embedding_model)
retriever = vectorstore.as_retriever()

#prompts
system_prompt = SystemMessagePromptTemplate.from_template("""
    You are a chatbot to answer about {codebase} which is written in
    the {language} programming language.  {attitude}
""")
human_prompt = HumanMessagePromptTemplate.from_template("{query}.  Use this as {content}")

#print greeintgs
attitude = "Be brief and ask how can I help you?"
content = "FMSCoupler is the GFDL coupling program"
query = "Introduce yourself"
messages = [
    system_prompt.format(language="Fortran", codebase="FMSCoupler", attitude=attitude),
    human_prompt.format(query=query, content=content),
]

query = "hello!"
while "goodbye" not in query.lower():

    query = input(llama3.invoke(messages).content+"\n> ")

    responses = retriever.invoke(query)

    messages = [
        system_prompt.format(language="Fortran", codebase="FMSCoupler", attitude="brief"),
        human_prompt.format(query=query, content=responses),
    ]







