import logging
from datetime import datetime
from typing import Any

import yaml
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

logger: logging.Logger = logging.getLogger(__name__)

OLLAMA_CHAT_MODEL = "mistral-nemo:latest"
HYBRID_LIMIT = 24
OUTPUT_LOG_FILE = "chatbot_output_log.yaml"

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
        

    def simple_retrieve(self, question: str) -> list[tuple[Document, float]]:
        """Search unified vectorstore and assemble sibling chunks by parent."""

        logger.debug(f"Querying vectorstore with top_k={HYBRID_LIMIT}: '{question}'")

        docs_and_scores = self.vectorstore.similarity_search_with_score(
            question, k=HYBRID_LIMIT, **self.hybrid_kwargs
        )

        logger.debug(f"Raw similarity distance scores retrieved: {[score for _, score in docs_and_scores]}")

        return docs_and_scores

    def _log_output(self,
                     question: str,
                     answer: str,
                     docs_and_scores: list[tuple[Document, float]]) -> None:
        """Append a structured record of the exchange to the YAML output log."""

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "llm_model": self.model_name,
            "system_prompt": self.system_message,
            "user_query": question,
            "ai_response": answer,
            "retrieved_files": [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "similarity_score": score,
                }
                for doc, score in docs_and_scores
            ],
        }

        with open(OUTPUT_LOG_FILE, "a", encoding="utf-8") as log_file:
            yaml.safe_dump([log_entry], log_file, sort_keys=False, allow_unicode=True)
            log_file.write("\n")

    def ask(self, question: str) -> tuple[str, list[tuple[Document, float]], str]:
        """Invoke"""
        docs_and_scores = self.retrieve(question)
        context = "\n\n".join([doc.page_content for doc, _ in docs_and_scores])
        answer = self.answer_chain.invoke({"question": question, "context": context})
        self._log_output(question, answer, docs_and_scores)
        return answer, docs_and_scores, context
