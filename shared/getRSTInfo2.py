import logging
from datetime import datetime

import yaml
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, UnstructuredRSTLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

logging.basicConfig(
    filename="catalog_bot_output_log.yaml",
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("catalog_bot")

EMBEDDING_MODEL_NAME = "llama3.2"
LLM_MODEL_NAME = "llama3.2"
PERSIST_DIRECTORY = "./catalog_bot_db"
DOCS_DIRECTORY = "../doc"
DEFAULT_QUERY = "Do I have to use fre-cli to generate a catalog?"

SYSTEM_PROMPT = (
    "You are a professional assistant for the Catalog Builder tool. "
    "Use the provided context to answer the user's question accurately. "
    "If the answer isn't in the context, say you don't know—don't guess. "
    "If your answer is composed of separate thoughts be sure to separate them with ample space (but not too much) as to not confuse the reader. "
    "Remember that the package is not installable via PyPi. The recommended approach is to pip install locally, but the package is available via conda. "
    "Do not tell the user to click or scroll on anything. This is a CLI tool. "
    "Do not mention inventory, sales, or products in your responses - The tool is not meant to be used for that. "
    "Organize your response with clear steps or bullet points if needed.\n\n"
    "Context:\n{context}"
)


def log_output(llm_model, system_prompt, user_query, response_text, retrieved_docs):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "llm_model": llm_model,
        "system_prompt": system_prompt,
        "user_query": user_query,
        "ai_response": response_text,
        "retrieved_files": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "similarity_score": score,
            }
            for doc, score in retrieved_docs
        ],
    }

    with open("catalog_bot_output_log.yaml", "a", encoding="utf-8") as log_file:
        yaml.safe_dump([log_entry], log_file, sort_keys=False, allow_unicode=True)
        log_file.write("\n")


def load_rst_documents(doc_path):
    loader = DirectoryLoader(
        doc_path,
        glob="**/*.rst",
        loader_cls=UnstructuredRSTLoader,
        loader_kwargs={"mode": "elements"},
    )
    return loader.load()


def split_documents(raw_rst_docs):
    rst_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.RST,
        chunk_size=1000,
        chunk_overlap=150,
    )
    return rst_splitter.split_documents(raw_rst_docs)


def build_vectorstore(documents, embedding_model, persist_directory):
    embeddings = OllamaEmbeddings(model=embedding_model)
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )


def build_prompt(system_prompt):
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])


def retrieve_documents_with_scores(vectorstore, user_query, k=5):
    return vectorstore.similarity_search_with_relevance_scores(user_query, k=k)


def build_context(retrieved_docs):
    return "\n\n".join(doc.page_content for doc, _score in retrieved_docs)


def generate_response(llm, prompt, user_query, context):
    messages = prompt.format_messages(
        input=user_query,
        context=context,
    )
    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)


def print_load_summary(raw_rst_docs, rst_chunks):
    print(f"Loaded {len(raw_rst_docs)} source files and created {len(rst_chunks)} chunks.")
    sources = {doc.metadata["source"] for doc in raw_rst_docs}
    print(f"Loaded files from: {sources}")


def main():
    raw_rst_docs = load_rst_documents(DOCS_DIRECTORY)
    rst_chunks = split_documents(raw_rst_docs)
    print_load_summary(raw_rst_docs, rst_chunks)

    vectorstore = build_vectorstore(
        documents=rst_chunks,
        embedding_model=EMBEDDING_MODEL_NAME,
        persist_directory=PERSIST_DIRECTORY,
    )

    llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0)
    prompt = build_prompt(SYSTEM_PROMPT)

    user_query = DEFAULT_QUERY
    retrieved_docs = retrieve_documents_with_scores(vectorstore, user_query, k=5)
    context = build_context(retrieved_docs)
    response_text = generate_response(llm, prompt, user_query, context)

    log_output(
        llm_model=LLM_MODEL_NAME,
        system_prompt=SYSTEM_PROMPT,
        user_query=user_query,
        response_text=response_text,
        retrieved_docs=retrieved_docs,
    )

    print(response_text)


if __name__ == "__main__":
    main()
