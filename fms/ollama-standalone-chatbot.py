# llama_index based chatbot that uses the GithubReader to parse code
# to run must have a token set up for pulling from github, stored as GITHUB_TOKEN

import os
import sys
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.core.instrumentation import get_dispatcher
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import TokenTextSplitter

# Set up event handler
class GitHubEventHandler(BaseEventHandler):
    def handle(self, event):
        if isinstance(event, GitHubFileProcessedEvent):
            print(f"Processed file: {event.file_path}")

# 1. Validate GitHub Token
if "GITHUB_TOKEN" not in os.environ:
    print("Error: Please set the GITHUB_TOKEN environment variable.")
    sys.exit(1)

# 2. Configure LlamaIndex to use Local Ollama Models globally
print("Initializing local LLM and Embedding models...")
Settings.llm = Ollama(model="qwen2.5-coder:7b",context_window=8192, request_timeout=300.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)

# Configuration for the target repository
PERSIST_DIR = "./local_storage"
GITHUB_OWNER = "noaa-gfdl"
GITHUB_REPO = "fms"
GITHUB_BRANCH = "main"

def get_codebase_index():
    """Loads local index or fetches from GitHub and indexes using local embeddings."""
    if os.path.exists(PERSIST_DIR):
        print("Loading existing local index from storage...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    else:
        print(f"Fetching repository {GITHUB_OWNER}/{GITHUB_REPO} from GitHub...")
        dispatcher = get_dispatcher()
        handler = GitHubEventHandler()
        dispatcher.add_event_handler(handler)

        github_client = GithubClient(
           github_token=os.environ["GITHUB_TOKEN"],
           verbose=True
        )
        reader = GithubRepositoryReader(
           github_client=github_client,
           owner="noaa-gfdl",
           repo="fms",
        )
        # Load all files from a branch
        branch_documents = reader.load_data(branch="main")
        index = VectorStoreIndex.from_documents(branch_documents)
        
        documents = reader.load_data(branch=GITHUB_BRANCH)
        print(f"Loaded {len(documents)} documents. Generating local embeddings...")
        
        # This will use the local Nomic embedding model set in Settings
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print(f"Local index successfully saved to {PERSIST_DIR}")
        
    return index

def main():
    # grabs the code from github and creates a VectorStore
    index = get_codebase_index()
    # create the chatbot from the read in code, uses Settings class for config 
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context", 
        verbose=False
    )
    print("\n" + "="*50)
    print(f"Local Chatbot ready! Asking questions using local Ollama models.")
    print("Type 'exit' or 'quit' to end the session.")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
                
            if not user_input.strip():
                continue
            
            print("\nBot is thinking (running locally)...")
            response = chat_engine.chat(user_input)
            print(f"\nBot: {response}\n")
            print("-" * 30)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
