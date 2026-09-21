import logging
from datetime import datetime
from typing import Any

import yaml
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

OLLAMA_CHAT_MODEL = "mistral-nemo:latest"
HYBRID_LIMIT = 24
OUTPUT_LOG_FILE = "chatbot_output_log.yaml"

# Global state for YAML logging
_yaml_initialized = False


def _initialize_yaml_log(model_name: str, system_message: str):
    """Initialize the YAML log file with model and system prompt at top level."""
    global _yaml_initialized
    if _yaml_initialized:
        return
    
    header = {
        "model": model_name,
        "system_prompt": system_message,
    }
    
    with open(OUTPUT_LOG_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(header, f, sort_keys=False, allow_unicode=True)
    
    _yaml_initialized = True


def _log_interaction(question: str, response: str, docs_and_scores: list[tuple[Document, float]]):
    """Append a structured Q&A interaction to the YAML output log."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": question,
        "response": response,
        "retrieved_files": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "similarity_score": float(score),
            }
            for doc, score in docs_and_scores
        ],
    }
    
    with open(OUTPUT_LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write("\n- " + yaml.dump(log_entry, sort_keys=False, allow_unicode=True)[2:])


class RAGChatbot:

    def __init__(self,
                 vectorstore: Any,
                 system_message: str, 
                 temperature: float = 0,
                 model_name: str = OLLAMA_CHAT_MODEL,
                 search_hybrid = False,
                 retrieve_function: Any = None
    ):
        
        self.chatbot = ChatOllama(model=model_name, temperature=temperature)
        self.model_name = model_name

        # Load vectorstore
        self.vectorstore = vectorstore

        # Set up hybrid search if requested
        self.search_hybrid = search_hybrid
        self.hybrid_kwargs = {"ranker_type": "rrf", "ranker_params": {"k": 60}} if search_hybrid else {}

        # Custom retrieve function
        # Custom retrieve functions must return a list of (doc, score)
        if retrieve_function is None:
            self.retrieve = self.simple_retrieve
        else:
            self.retrieve = retrieve_function
            
        self.system_message = system_message

        self.prompt = ChatPromptTemplate.from_messages(
            [("system", self.system_message), ("human", "{question}")]
        )

        self.answer_chain = self.prompt | self.chatbot | StrOutputParser()
        

        # Initialize YAML logging if enabled
        if enable_yaml_logging:
            configure_yaml_logging(log_file)
            _initialize_yaml_log(model_name, system_message)
        
        self.enable_yaml_logging = enable_yaml_logging


    def simple_retrieve(self, question: str) -> list[tuple[Document, float]]:
        """Search unified vectorstore and assemble sibling chunks by parent."""

        docs_and_scores = self.vectorstore.similarity_search_with_score(
            question, k=HYBRID_LIMIT, **self.hybrid_kwargs
        )

        return docs_and_scores

    def ask(self, question: str) -> tuple[str, list[tuple[Document, float]], str]:
        """Invoke"""
        docs_and_scores = self.retrieve(question)
        context = "\n\n".join([doc.page_content for doc, _ in docs_and_scores])
        answer = self.answer_chain.invoke({"question": question, "context": context})
        
        # Log interaction to YAML if enabled
        if self.enable_yaml_logging:
            _log_interaction(question, answer, docs_and_scores)
        
        return answer, docs_and_scores, context
