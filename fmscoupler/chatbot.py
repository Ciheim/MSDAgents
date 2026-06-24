from document_utils import (
    connect_vectorstore,
)
from shared.chatbot import RAGChatbot

OLLAMA_CHAT_MODEL = "mistral-nemo:latest"
COLLECTION_NAME = "FMSCoupler"

system_message = """
You are a technical assistant for the FMSCoupler.  

## Background
FMSCoupler, Flexible Modeling Systems Coupler, is a set of program
and modules to couple the atmosphere, ocean, land, and ice components in the 
GFDL (Geophysical Fluid Dynamics Laboratory) coupled climate models.

## Instructions:
- Use only the supplied context to answer.
- If the context does not contain the answer, say you do not know.
- Answer with a concise explanation.  Do not use markdown formatting.

## Context: 
{context}
"""

chatbot = RAGChatbot(
    vectorstore=connect_vectorstore(COLLECTION_NAME), 
    system_message=system_message,
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