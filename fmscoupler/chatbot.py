from document_utils import (
    connect_vectorstore,
)
from shared.chatbot import RAGChatbot

OLLAMA_CHAT_MODEL = "mistral-nemo:latest"
COLLECTION_NAME = "FMSCoupler"

codebase_description = """
FMSCoupler.  FMSCoupler, Flexible Modeling Systems Coupler, is a set of program
and modules to couple the atmosphere, ocean, land, and ice components in the 
GFDL (Geophysical Fluid Dynamics Laboratory) coupled climate models.
"""

chatbot = RAGChatbot(
    vectorstore=connect_vectorstore(COLLECTION_NAME), 
    codebase_description=codebase_description,
    model_name=OLLAMA_CHAT_MODEL, 
)

while True:
    user_question = input("\nYou: ").strip()
    if user_question.lower() in {"quit", "exit", "q"}:
        print("Bye.")
        break

    response, docs_and_scores = chatbot.ask(user_question)        
    sources = ", ".join([doc.metadata.get("source")+"/"+doc.metadata.get("name") for doc, _ in docs_and_scores])
    print(f"\nAssistant: {response}")
    print(f"Relevant sources: {sources}")
    print("\n\n")