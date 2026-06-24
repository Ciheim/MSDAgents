from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import GitLoader
from langchain_ollama import OllamaEmbeddings

LOCAL_DB_PATH="./.fms_langchain_db"

def text_file_filter(file_path: str) -> bool:
    # Ignore anything in the hidden .git directory completely
    if ".git/" in file_path or ".github/" in file_path:
        return False
    # Only read standard code/text extensions
    return file_path

# Initialize the loader
loader = GitLoader(
    clone_url="https://github.com/noaa-gfdl/fms",
    repo_path="./.fms_clone/",  # Local directory to clone into
    branch="main",                 # Optional: defaults to master/main
    file_filter=text_file_filter,
)

# Load the repository files into LangChain Documents
documents = loader.load()

# Inspect the data
print(f"Loaded {len(documents)} files.")
print(f"First file content snippet:\n{documents[0].page_content[:100]}")
print(f"First file metadata: {documents[0].metadata}")

# 1. Split the code into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# 2. Embed and store in a local Vector DB using Ollama embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=LOCAL_DB_PATH)
retriever = vector_store.as_retriever()

# 3. Set up your local Ollama LLM
llm = Ollama(model="llama3")

# 4. Create the QA Chain
system_prompt = (
    "You are an expert developer. Use the given repository context to answer "
    "the question. If you don't know, say so.\n\nContext:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 5. Ask a question!

while True:
    try:
        user_input = input("You: ")
        if user_input.strip().lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not user_input.strip():
            continue
        
        print("\nBot is thinking (running locally)...")
        response = rag_chain.invoke({"input": user_input.strip()})
        print(response["answer"])
        print("-" * 30)
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
