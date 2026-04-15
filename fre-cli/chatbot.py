import chromadb

#import langchain
from langchain.agents import create_agent
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
    """
)

human_message = HumanMessagePromptTemplate.from_template(
    """
    {query}.  
    Only use this content to answer the question: {content}.
    Do not add any external knowledge.
    """
)

#agent_prompt = """
#Only answer questions about fre make.  
#You have one tool available called example_tool.  
#Call example_tool when ever the user says fremorizer.
#Else, use the content provided to answer the question
#"""

#def example_tool():
#    """
#    Print "please rename me" whenever the user says 'fremorizer'
#    """
#    print("please rename me")

#def another_example_tool():
#    """
#    To be called when the user asks about fre make
#    """
#    return collections[0].query(query_texts=[query])["documents"]


#load database
client = chromadb.PersistentClient(db_path)
collections = client.list_collections()

#create chatbot 
chatbot = ChatOllama(model=llm, temperature=0.0)
#agent = create_agent(chatbot, tools=[example_tool], system_prompt=agent_prompt)

query = "Say hello"
print(f"{chatbot.invoke(query).content}\n>", end=" ")
# agent.invoke({"messages":[human_message.format(query="Introduce yourself", content="")]})

while True:
    query = input()
    if "goodbye" in query.lower(): exit        
    content = collections[0].query(query_texts=[query])["documents"]
    answer = chatbot.invoke([system_message.format(), human_message.format(query=query, content=content)])
    #agent.invoke({"messages":[human_message.format(query=query, content=content)]})
    print(answer.content + "\n >", end=" ")

