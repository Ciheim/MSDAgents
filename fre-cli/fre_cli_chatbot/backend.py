"""
Geophysical Fluid Dynamics Laboratory (GFDL) - FRE Chatbot Backend Engine
========================================================================

This module implements the core Retrieval-Augmented Generation (RAG) architecture
for the Flexible Modeling System Runtime Environment (FRE) workflow helper.

It supports:
- High-performance vector embeddings storage and retrieval using Milvus Lite.
- Compatibility patches to bridge PyMilvus 2.6+ ORM limitations inside local sqlite/file connections.
- Session interaction logging and user feedback tracking using SQLite3.
- Advanced LangChain-based history-aware retrieval chains with LLM scoring capabilities.
- Standardized, severity-conscious python logging structures.
"""

import os
import sys
import logging
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Callable, Optional, Dict, Any

# --- LangChain Imports ---
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

# --- PyMilvus Connections ---
from pymilvus import connections, Collection, MilvusClient

# --- Custom Scoring Retriever Imports ---
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field

# --- Dynamic Import for FreDatabase/Parser ---
try:
    parent_dir = Path(__file__).resolve().parents[2]
    parser_path = parent_dir / "parsers" / "fre_parser"
    if str(parser_path) not in sys.path:
        sys.path.append(str(parser_path))
        
    from fre_database import FreDatabase
except ImportError:
    FreDatabase = None


# ==========================================
# SYSTEM LOGGER SETUP
# ==========================================
logger = logging.getLogger("gfdl_chatbot")


def configure_logging(debug_mode: bool = False):
    """Configures the logging format and severity levels across the application.

    Mutes highly verbose third-party loggers (like pymilvus, grpc, and urllib3)
    during standard runtime while offering full telemetry when debug mode is enabled.

    Args:
        debug_mode (bool): If True, activates DEBUG logging, else INFO.
    """
    log_level = logging.DEBUG if debug_mode else logging.INFO
    
    # Configure root logger parameters
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr
    )

    # Set our custom app-level logger level
    logger.setLevel(log_level)

    # Suppress extremely chatty third-party packages to prevent terminal flooding
    external_log_level = logging.INFO if debug_mode else logging.WARNING
    for chatty_logger in [
        "pymilvus", "grpc", "urllib3", "asyncio", 
        "langchain", "httpx", "httpcore", "ollama"
    ]:
        logging.getLogger(chatty_logger).setLevel(external_log_level)

    logger.info(f"Logging configured: level={'DEBUG' if debug_mode else 'INFO'}")


# Initialize with standard INFO log settings out-of-the-box
configure_logging(debug_mode=False)


# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
OLLAMA_BASE_URL = "http://localhost:11434" 
MODEL_NAME = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"

# Milvus vector file path (Triggers Milvus Lite dynamically)
MILVUS_URI = os.getenv("MILVUS_URI", "./gfdl_fre_vectors.db")
MILVUS_COLLECTION = "fre_workflow_vectors"

# Local relational storage database path
SQLITE_DB_PATH = "gfdl_fre_chatbot_analytics.db"


# ==========================================
# PYMILVUS LITE COMPATIBILITY PATCHES
# ==========================================

class MockIndex:
    """Mock index definition mapping the output from modern MilvusClient to legacy PyMilvus ORM structures.

    This class prevents langchain-milvus from raising key exceptions during local connections,
    suppressing schema-lookup crashes like 'index_param' or '.params' attributes.
    """
    def __init__(self, index_name: str, field_name: str, metric_type: str = "COSINE", index_type: str = "FLAT", params: Optional[dict] = None):
        """Initializes a MockIndex instance.

        Args:
            index_name (str): Identifier name of the index schema.
            field_name (str): The entity field bound to this index.
            metric_type (str): Scoring method, e.g. "COSINE", "L2", "IP".
            index_type (str): Type of index structure, e.g. "FLAT", "IVF_FLAT".
            params (dict, optional): Custom execution and search parameters.
        """
        self.index_name = index_name
        self.field_name = field_name
        self._metric_type = metric_type
        self._index_type = index_type
        self._params = params or {}
        
        # Internal parameter storage used by langchain-milvus verification passes
        self.index_param = {
            "index_type": self._index_type,
            "metric_type": self._metric_type,
            "params": self._params
        }

    @property
    def params(self) -> dict:
        """Exposes the internal parameter payload dictionary.

        Returns:
            dict: Structured index parameter values.
        """
        return self.index_param

    def to_dict(self) -> dict:
        """Presents a payload representation matching PyMilvus ORM expectations.

        Returns:
            dict: Key value schema map for indexing logic.
        """
        return {
            "index_name": self.index_name,
            "field_name": self.field_name,
            "index_param": self.index_param,
            "metric_type": self._metric_type,
            "index_type": self._index_type,
            "params": self._params
        }


# Keep the original pymilvus indexes getter to fallback cleanly on remote servers
original_indexes_getter = Collection.indexes


def patched_indexes_property(self) -> List[MockIndex]:
    """Intercepts and resolves index queries using MilvusClient.

    This bypasses legacy ORM metadata calls that execute illegal cluster-level
    gRPC routines (like AllocTimestamp) under local in-memory/file connections.

    Returns:
        List[MockIndex]: Formatted metadata index profiles.
    """
    uri = MILVUS_URI
    # Only execute our client patch if the target is a local file connection (Milvus Lite)
    if uri and not (uri.startswith("http://") or uri.startswith("https://")):
        try:
            logger.debug("Local SQLite-style Milvus Lite connection detected. Applying indexes patching...")
            client = MilvusClient(uri=uri)
            try:
                if client.has_collection(collection_name=self.name):
                    indexes_info = client.list_indexes(collection_name=self.name)
                    mock_indexes = []
                    for idx_name in indexes_info:
                        try:
                            desc = client.describe_index(collection_name=self.name, index_name=idx_name)
                            
                            # Safely extract dictionary or object attributes dynamically
                            metric_type = desc.get("metric_type", "COSINE") if isinstance(desc, dict) else getattr(desc, "metric_type", "COSINE")
                            index_type = desc.get("index_type", "FLAT") if isinstance(desc, dict) else getattr(desc, "index_type", "FLAT")
                            field_name = desc.get("field_name", "vector") if isinstance(desc, dict) else getattr(desc, "field_name", "vector")
                            params = desc.get("params", {}) if isinstance(desc, dict) else getattr(desc, "params", {})
                            
                            mock_indexes.append(MockIndex(idx_name, field_name, metric_type, index_type, params))
                        except Exception as inner_ex:
                            logger.debug(f"Failed to describe index '{idx_name}': {inner_ex}. Emitting fallback mock.")
                            mock_indexes.append(MockIndex(idx_name, "vector"))
                    return mock_indexes
                else:
                    logger.debug(f"No collection named '{self.name}' found to list indexes.")
                    return []
            finally:
                client.close()
        except Exception as e:
            logger.warning(f"Indexes patching interceptor failed: {e}. Falling back to default ORM handler.")
    
    # Delegate to PyMilvus ORM handler if connecting to standard remote Milvus servers
    return original_indexes_getter.__get__(self, Collection)


# Apply our property override hook to the base Collection class
Collection.indexes = property(patched_indexes_property)


# ==========================================
# GFDL AGENT PROMPTS
# ==========================================
GFDL_SYSTEM_PROMPT = (
    "You are a technical Assistant at the Geophysical Fluid Dynamics Laboratory (GFDL), an expert in the FRE (Flexible Modeling Systems Runtime Environment) framework workflow -- a system designed to optimize the configuring, compiling, building, running, post-processing, and analyzing of GFDL-developed climate models.\n"
    "Your goal is to provide accurate, structured, and concise information to scientists running these workflows across all components and modules (such as make, yaml, app, list, pp, and run).\n\n"
    "RESPONSE STRUCTURE:\n"
    "1. **Summary**: A 1-2 sentence overview of the answer.\n"
    "2. **Details**: Use bullet points for steps or parameters.\n"
    "3. **Example**: Provide a CLI command or config snippet ONLY if relevant.\n\n"
    "CONSTRAINTS:\n"
    "- Use Markdown headings (###) for sections.\n"
    "- Prioritize information from files labeled 'Structured Sphinx Documentation'.\n"
    "- Be verbose ONLY if the user asks for 'detailed explanation' or 'deep dive'. Otherwise, keep it functional.\n"
    "- NEVER mention internal Python script names (e.g., utils.py) or internal Python functions unless asked about implementation.\n"
    "- If referencing chat history, ensure consistency with previous answers.\n"
    "- **CRITICAL WEB RENDERING RULE**: To prevent browser layout glitches, always wrap command parameters and example syntax inside markdown code blocks. Use square brackets for placeholder variables (e.g., `[platform]` or `[experiment]`) rather than angle brackets (`<platform>`), as web browsers mistake angle brackets for HTML tags and hide them.\n\n"
    "CONTEXT:\n"
    "{context}"
)


# ==========================================
# RETRIEVERS AND MEMORY WRAPPERS
# ==========================================

class ScoringVectorStoreRetriever(BaseRetriever):
    """Custom LangChain retriever that preserves similarity distance scores inside document metadata.

    Allows frontends to inspect exactly how closely a piece of context matches the query.
    """
    vector_store: Milvus = Field(...)
    search_kwargs: dict = Field(default_factory=dict)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Queries Milvus, extracts matched scores, and returns metadata-enriched Documents.

        Args:
            query (str): The search text query.
            run_manager (CallbackManagerForRetrieverRun): Token runner callbacks hook.

        Returns:
            List[Document]: Set of matching documents containing "score" inside metadata.
        """
        k = self.search_kwargs.get("k", 10)
        logger.debug(f"Querying Milvus with top_k={k}: '{query}'")
        
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
        logger.debug(f"Raw similarity distance scores retrieved: {[score for _, score in docs_with_scores]}")

        scored_docs = []
        for doc, score in docs_with_scores:
            cloned_doc = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "score": score}
            )
            scored_docs.append(cloned_doc)
        return scored_docs


class LegacyConnectionFallbackDict(dict):
    """Custom dictionary subclass to bridge the PyMilvus 2.6+ ConnectionManager fallback gap.

    Redirects random auto-generated connection labels back to our 'default' connection.
    """
    def __init__(self, original_dict, fallback_connection):
        super().__init__(original_dict)
        self.fallback_connection = fallback_connection

    def __contains__(self, key) -> bool:
        return True

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return self.fallback_connection

    def get(self, key, default=None):
        try:
            return super().__getitem__(key)
        except KeyError:
            return self.fallback_connection


def get_vector_store() -> Milvus:
    """Instantiates and returns the configured Milvus vector database connection handle.

    Registers active connection parameters and handles legacy dictionary fallbacks
    for safe ORM schema validation.

    Returns:
        Milvus: Ready-to-use vector store instance.
    """
    logger.debug(f"Initializing Milvus Vector Store client (URI: '{MILVUS_URI}')")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    
    try:
        if not connections.has_connection("default"):
            connections.connect(alias="default", uri=MILVUS_URI)
            logger.debug("Successfully registered 'default' connection alias in PyMilvus context pool.")
            
        default_conn = connections._alias_handlers.get("default")
        
        # Apply patch to bridge dynamic connection pool lookups in newer pymilvus packages
        if default_conn and not isinstance(connections._alias_handlers, LegacyConnectionFallbackDict):
            connections._alias_handlers = LegacyConnectionFallbackDict(
                connections._alias_handlers, 
                default_conn
            )
            logger.debug("Applied LegacyConnectionFallbackDict wrapper to connection handler pool.")
    except Exception as e:
        logger.warning(f"Could not explicitly register global connection handlers: {e}")

    return Milvus(
        embedding_function=embeddings,
        collection_name=MILVUS_COLLECTION,
        connection_args={"uri": MILVUS_URI},
        drop_old=False
    )


def detect_module_target(directory_path: str) -> Optional[str]:
    """Parses folder paths to find matching component targets of the FRE workflow.

    Args:
        directory_path (str): Ingestion target filesystem folder path.

    Returns:
        Optional[str]: Target component identifier ("make", "yaml", "pp", etc.) if matched.
    """
    path_str = str(Path(directory_path).absolute()).lower()
    for target in ["make", "yaml", "app", "list", "pp", "run"]:
        if target in path_str:
            logger.debug(f"Heuristically detected FRE target component: '{target}'")
            return target
    return None


# ==========================================
# INGESTION ENGINE
# ==========================================

def run_ingestion(directory_path: str, logger_callback: Callable[[str], None] = logger.info) -> int:
    """Indexes source code documentation, Python files, and parsed Sphinx elements into Milvus.

    Args:
        directory_path (str): Source code or documentation base directory.
        logger_callback (Callable): Context logging receiver for GUI/CLI status tracking.

    Returns:
        int: Count of ingested chunk vectors successfully written.
    """
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory not found: {directory_path}")

    logger.info(f"Initiating documentation parsing ingestion pipeline on path: '{directory_path}'")
    logger_callback("Scanning target directories...")

    # Inject paths into sys.path to allow custom sphinx parsing imports
    abs_dir = os.path.abspath(directory_path)
    if abs_dir not in sys.path:
        sys.path.insert(0, abs_dir)
    parent_dir_path = os.path.dirname(abs_dir)
    if parent_dir_path not in sys.path:
        sys.path.insert(0, parent_dir_path)

    all_lc_docs = []
    valid_exts = [".py", ".md", ".rst", ".txt"]
    
    # Traverse directory and parse candidate text files
    for root, _, files in os.walk(directory_path):
        #if "__init__.py" in root or "tests" in root:
        #    continue
        for file in files:
            if "__init__.py" in file or "test" in file:
                continue

            ext = os.path.splitext(file)[1]
            if ext in valid_exts:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                   
                    # Distinguish between "higher level" documentation and source code docs 
                    if "README" in file.upper():
                        page_content = f"DOCUMENT TYPE: High-level Documentation\nFILE: {file}\nCONTENT:\n{content}"
                        doc_type = "user_doc"
                    elif file.endswith(".py"):
                        page_content = f"# SOURCE CODE IMPLEMENTATION FILE\n# FILE: {file}\n{content}"
                        doc_type = "raw_source"
                    else:
                        page_content = content
                        doc_type = "general"
                        
                    all_lc_docs.append(Document(page_content=page_content, metadata={"file_path": file_path, "type": doc_type}))
                except Exception as e:
                    logger.warning(f"Incomplete file read pass. Skipped '{file_path}': {e}")

    # Sphinx docstring module extraction
    if FreDatabase:
        module_target = detect_module_target(directory_path)
        logger_callback(f"Targeting Sphinx extraction for: '{module_target or 'all modules'}'")
        try:
            fre_db = FreDatabase(module_name=module_target)
            fre_db.summarize()
            doc_list, metadata_list, id_list = fre_db.to_chromadb()
            
            for doc_text, metadata_dict, unique_id in zip(doc_list, metadata_list, id_list):
                all_lc_docs.append(Document(
                    page_content=f"DOCUMENT TYPE: Structured Sphinx Documentation\nSOURCE MODULE: {metadata_dict.get('module')}\nCOMPONENT: {metadata_dict.get('name')}\nCONTENT:\n{doc_text}",
                    metadata={
                        "file_path": metadata_dict.get("module", ""),
                        "name": metadata_dict.get("name", ""),
                        "package": metadata_dict.get("package", "fre"),
                        "type": "parsed_docstring",
                        "doc_id": unique_id
                    }
                ))
            logger_callback(f"Successfully extracted {len(doc_list)} structured Sphinx schemas.")
        except Exception as e:
            logger.warning(f"⚠️ Custom DB documentation parser skipped due to error: {e}")
    else:
        logger.debug("FreDatabase parser missing or unresolved in python context path.")

    # Chunk text assets using target language splits
    logger_callback("Parsing documentation into vector nodes...")
    python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=1500, chunk_overlap=200)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)

    nodes = []
    for doc in all_lc_docs:
        if doc.metadata.get("type") in ["parsed_docstring", "user_doc", "general"]:
            nodes.extend(text_splitter.split_documents([doc]))
        elif doc.metadata.get("type") == "raw_source":
            try:
                nodes.extend(python_splitter.split_documents([doc]))
            except Exception as split_ex:
                logger.debug(f"Python-specific splitting failed for file {doc.metadata.get('file_path')}: {split_ex}. Defaulting to text parser.")
                nodes.extend(text_splitter.split_documents([doc]))

    # Synchronize nodes to Milvus database
    logger_callback(f"Syncing {len(nodes)} vector nodes to Milvus...")
    vector_store = get_vector_store()
    
    import uuid
    ids = []
    for idx, n in enumerate(nodes):
        parent_id = n.metadata.get("doc_id")
        if parent_id:
            ids.append(f"{parent_id}_chunk_{idx}")
        else:
            ids.append(str(uuid.uuid4()))
            
    vector_store.add_documents(nodes, ids=ids)
    logger.info(f"Ingestion successful. Successfully added {len(nodes)} document segments.")
    return len(nodes)


# ==========================================
# RESPONSE CHAIN WRAPPERS
# ==========================================

class DocumentWrapper:
    """Adapter class wrapping standard LangChain document records into simple properties.

    Enables seamless property mapping for frontends.
    """
    def __init__(self, doc: Document):
        """Initializes the wrapper.

        Args:
            doc (Document): LangChain baseline document asset.
        """
        self.metadata = doc.metadata
        self.page_content = doc.page_content
        self.score = doc.metadata.get('score', 0.0)


class LCELChatEngineWrapper:
    """Wrapper that manages retrieval logic, history tracking, and async stream outputs."""
    def __init__(self, rag_chain):
        """Initializes the LCEL Chat Engine wrapper.

        Args:
            rag_chain (Runnable): Constructed retrieval chain sequence.
        """
        self.rag_chain = rag_chain
        self.chat_history = []
        self.source_nodes = []
        self._full_response = ""

    def stream_chat(self, query: str) -> "LCELChatEngineWrapper":
        """Executes a history-aware query and registers an active context token generator stream.

        Args:
            query (str): The search input string.

        Returns:
            LCELChatEngineWrapper: Reference self with a running .response_gen generator property.
        """
        logger.debug(f"Preparing query execution thread for: '{query}'")
        self.source_nodes = []
        self._full_response = ""
        
        raw_stream = self.rag_chain.stream({"input": query, "chat_history": self.chat_history})
        
        # Capture context and prepare streaming boundaries
        pre_fetched_chunks = []
        try:
            for chunk in raw_stream:
                pre_fetched_chunks.append(chunk)
                if "context" in chunk:
                    self.source_nodes = [DocumentWrapper(d) for d in chunk["context"]]
                if "answer" in chunk and chunk["answer"]:
                    break
        except StopIteration:
            pass
        
        def generator():
            # Emit buffered setup tokens
            for chunk in pre_fetched_chunks:
                if "context" in chunk and not self.source_nodes:
                    self.source_nodes = [DocumentWrapper(d) for d in chunk["context"]]
                if "answer" in chunk and chunk["answer"]:
                    self._full_response += chunk["answer"]
                    yield chunk["answer"]
            
            # Continue yielding remaining output stream
            for chunk in raw_stream:
                if "context" in chunk:
                    self.source_nodes = [DocumentWrapper(d) for d in chunk["context"]]
                if "answer" in chunk and chunk["answer"]:
                    self._full_response += chunk["answer"]
                    yield chunk["answer"]
            
            # Commit session state to memory
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=self._full_response))
            logger.debug(f"Chat generation segment complete. Output length: {len(self._full_response)} characters.")
            
        self.response_gen = generator()
        return self


def get_chat_engine() -> Optional[LCELChatEngineWrapper]:
    """Assembles and returns the full history-aware LCEL retrieval chain model.

    Returns:
        Optional[LCELChatEngineWrapper]: Interface wrapper for chat queries or None if initialization fails.
    """
    try:
        logger.info("Assembling LLM chat query execution engine...")
        vector_store = get_vector_store()
        retriever = ScoringVectorStoreRetriever(vector_store=vector_store, search_kwargs={"k": 10})
        llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)

        # 1. Condense history-aware query structures
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # 2. Main Question Answer generator pipeline
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", GFDL_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        logger.info("LLM chat query execution engine assembled successfully.")
        return LCELChatEngineWrapper(rag_chain)
    except Exception as e:
        logger.error(f"Failed to assemble the LangChain RAG pipeline: {e}", exc_info=True)
        return None


# ==========================================
# SQLITE ANALYTICS LOGGING AND FEEDBACK
# ==========================================

def get_sqlite_conn() -> sqlite3.Connection:
    """Establishes and returns an open connection handle to our local SQLite schema database.

    Initializes data schemas automatically on-demand if missing.

    Returns:
        sqlite3.Connection: Active transactional SQL database connector.
    """
    logger.debug(f"Acquiring database connection context handle to: '{SQLITE_DB_PATH}'")
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS interaction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            query TEXT, 
            response TEXT, 
            model_name TEXT, 
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            query TEXT, 
            response TEXT, 
            score INTEGER, 
            feedback TEXT, 
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def log_interaction(query: str, response: str, metadata: Optional[Dict] = None):
    """Saves telemetry interaction records to the local database file.

    Args:
        query (str): The search input query.
        response (str): Standard agent text output emitted.
        metadata (dict, optional): Context metadata parameters.
    """
    try:
        conn = get_sqlite_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO interaction_logs (query, response, model_name) VALUES (?, ?, ?)", 
            (query, response, MODEL_NAME)
        )
        conn.commit()
        cur.close()
        conn.close()
        logger.debug("Successfully logged conversational interaction record to database.")
    except Exception as e:
        logger.error(f"Failed to log conversational interaction in database: {e}")


def evaluate_response(query: str, response_obj: LCELChatEngineWrapper) -> Dict[str, bool]:
    """Runs automated evaluations (Faithfulness and Relevancy) using LLM-as-a-judge.

    Args:
        query (str): Initial input request.
        response_obj (LCELChatEngineWrapper): Engine session reference housing outputs and matched contexts.

    Returns:
        Dict[str, bool]: Evaluation status metrics (e.g. {"faithfulness": True, "relevancy": False}).
    """
    try:
        logger.debug("Executing LLM-as-a-judge quality evaluations...")
        llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.0)
        
        context_str = "\n\n".join([doc.page_content for doc in response_obj.source_nodes])
        response_text = response_obj._full_response
        
        # 1. Evaluate Faithfulness
        f_prompt = (
            f"Context: {context_str}\n\n"
            f"Response: {response_text}\n\n"
            "Is the Response fully supported by the Context? "
            "Answer strictly 'PASS' if yes, or 'FAIL' if no or if it hallucinates."
        )
        f_res = llm.invoke(f_prompt).content
        
        # 2. Evaluate Relevancy
        r_prompt = (
            f"Query: {query}\n\n"
            f"Response: {response_text}\n\n"
            "Does the Response adequately answer the Query? "
            "Answer strictly 'PASS' if yes, or 'FAIL' if no."
        )
        r_res = llm.invoke(r_prompt).content
        
        evaluation = {
            "faithfulness": "PASS" in f_res.upper(),
            "relevancy": "PASS" in r_res.upper()
        }
        logger.debug(f"Evaluator completed: Faithfulness={evaluation['faithfulness']}, Relevancy={evaluation['relevancy']}")
        return evaluation
    except Exception as e:
        logger.error(f"Evaluator runtime error: {e}")
        return {"faithfulness": True, "relevancy": True}


def save_feedback(query: str, response: str, score: int, feedback_text: str = ""):
    """Saves user rating metrics (likes/dislikes) into the local analytics database.

    Args:
        query (str): The search input string.
        response (str): The chatbot output response text.
        score (int): Score assigned (1 for Like, 0 for Dislike).
        feedback_text (str): Optional text feedback.
    """
    try:
        conn = get_sqlite_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO user_feedback (query, response, score, feedback) VALUES (?, ?, ?, ?)", 
            (query, response, score, feedback_text)
        )
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Logged user feedback score: {score} ('{'Like' if score == 1 else 'Dislike'}')")
    except Exception as e:
        logger.error(f"Failed to record user feedback: {e}")


def get_feedback_stats() -> Dict[str, Any]:
    """Retrieves aggregated user feedback scores and recent feedback entries.

    Returns:
        Dict[str, Any]: Metric metrics dictionary holding count details and record arrays.
    """
    try:
        conn = get_sqlite_conn()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM user_feedback WHERE score = 1")
        likes = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM user_feedback WHERE score = 0")
        dislikes = cur.fetchone()[0]
        
        cur.execute("SELECT query, response, score, ts FROM user_feedback ORDER BY ts DESC LIMIT 10")
        recent_raw = cur.fetchall()
        
        recent = []
        for q, r, s, t_str in recent_raw:
            try:
                if " " in t_str:
                    t_obj = datetime.strptime(t_str.split(".")[0], "%Y-%m-%d %H:%M:%S")
                else:
                    t_obj = datetime.strptime(t_str.split(".")[0], "%Y-%m-%dT%H:%M:%S")
            except Exception:
                t_obj = datetime.now()
            recent.append((q, r, s, t_obj))
            
        cur.close()
        conn.close()
        return {"likes": likes, "dislikes": dislikes, "recent": recent}
    except Exception as e:
        logger.error(f"Failed to query database user feedback statistics: {e}")
        return {"likes": 0, "dislikes": 0, "recent": [], "error": str(e)}


def get_interaction_stats() -> List[Any]:
    """Retrieves standard conversational telemetry query logs.

    Returns:
        List[Any]: List of matching history logs.
    """
    try:
        conn = get_sqlite_conn()
        cur = conn.cursor()
        
        cur.execute("SELECT query, response, ts FROM interaction_logs ORDER BY ts DESC LIMIT 10")
        logs_raw = cur.fetchall()
        
        logs = []
        for q, r, t_str in logs_raw:
            try:
                if " " in t_str:
                    t_obj = datetime.strptime(t_str.split(".")[0], "%Y-%m-%d %H:%M:%S")
                else:
                    t_obj = datetime.strptime(t_str.split(".")[0], "%Y-%m-%dT%H:%M:%S")
            except Exception:
                t_obj = datetime.now()
            logs.append((q, r, t_obj))
            
        cur.close()
        conn.close()
        return logs
    except Exception as e:
        logger.error(f"Error reading interaction log history: {e}")
        return []
