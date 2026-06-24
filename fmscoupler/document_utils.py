"""
document_utils.py

Shared configuration, embedding model, splitters, and helper functions
used by create_readmes.py, document_code.py,
and create_database.py.
"""

import logging

from pydantic import BaseModel
from pymilvus import MilvusClient, connections

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

MILVUS_HOST      = "localhost"
MILVUS_PORT      = 19530
HUGGINGFACE_MODEL = "sentence-transformers/all-mpnet-base-v2"
DENSE_DIM        = 768   # output dimension of all-mpnet-base-v2
CHUNK_OVERLAP    = 100   # character overlap between sub-chunks
MAX_TEXT_LEN     = 65535
MAX_ID_LEN       = 512

# ---------------------------------------------------------------------------
# Embedding model 
# ---------------------------------------------------------------------------

dense_ef = HuggingFaceEmbeddings(
    model_name=HUGGINGFACE_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
tokenizer = dense_ef._client.tokenizer
MAX_TOKEN_LENGTH = tokenizer.model_max_length

# ---------------------------------------------------------------------------
# Markdown splitters and chunkers
# ---------------------------------------------------------------------------

splitters = {
    # h1/h2/h3 — used by the readme parser
    "readme": MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=True,
        return_each_line=False,
    ),
    # h1/h2 — first pass of the code-module parser
    "module": MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2")],
        strip_headers=True,
        return_each_line=False,
    ),
    # h3 — second pass of the code-module parser
    "subroutine": MarkdownHeaderTextSplitter(
        headers_to_split_on=[("###", "h3")],
        strip_headers=True,
        return_each_line=False,
    ),
}

chunkers = {
    "flowchart": RecursiveCharacterTextSplitter(
        separators=[r"(?=Step \d+:)"],
        chunk_size=MAX_TOKEN_LENGTH * 3,
        chunk_overlap=0,
        is_separator_regex=True,
    ),
    "arguments": RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=MAX_TOKEN_LENGTH * 3,
        chunk_overlap=0,
    ),
    "intro": RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". "],
        chunk_size=MAX_TOKEN_LENGTH * 3,
        chunk_overlap=CHUNK_OVERLAP,
    ),
    "description": RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". "],
        chunk_size=MAX_TOKEN_LENGTH * 3,
        chunk_overlap=CHUNK_OVERLAP,
    ),
    "misc": RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". "],
        chunk_size=MAX_TOKEN_LENGTH * 3,
        chunk_overlap=CHUNK_OVERLAP,
    )
}

# ---------------------------------------------------------------------------
# ID 
# ---------------------------------------------------------------------------

def make_id(idstrings) -> str:
    """Build a hierarchical chunk ID: h1/h2/h3 (omits missing levels)."""
    return "/".join(a_id for a_id in idstrings if a_id)

# ---------------------------------------------------------------------------
# Chunk helper
# ---------------------------------------------------------------------------

def check_chunk_length(chunk, end_in_error=False):
    """Check the token length of a section and print a warning if it exceeds MAX_TOKEN_LENGTH."""
    ntokens = len(tokenizer.tokenize(chunk))
    if ntokens > MAX_TOKEN_LENGTH:
        message = f"Section token length of {ntokens} exceeds the maximum of {MAX_TOKEN_LENGTH}."
        if end_in_error:
            raise RuntimeError(message)
        else:
            print(message)
    return ntokens

# ---------------------------------------------------------------------------
# Milvus helpers
# ---------------------------------------------------------------------------

def connect_to_client(): 
    client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", alias="default")
    return client

def connect_vectorstore(collection_name: str) -> Milvus:
    """Return a Milvus handle attached to an existing collection."""
    connect_to_client()
    return Milvus(
        embedding_function=dense_ef,
        builtin_function=BM25BuiltInFunction(),
        vector_field=["dense", "sparse"],
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT, "alias": "default"},
        collection_name=collection_name,
    )

def remove_collection_if_exists(collection_name: str) -> None:
    """Drop a Milvus collection if it already exists."""
    client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", alias="default")
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        print(f"[milvus] Dropped existing collection: {collection_name}")


def create_milvus_database(documents: list, ids: list, collection_name: str) -> None:
    """Embed documents and (re)build a Milvus hybrid collection."""
    print(f"\n[milvus] Connecting to {MILVUS_HOST}:{MILVUS_PORT}")
    print(f"[milvus] Collection  : {collection_name}")
    print(f"[milvus] Embedding   : {HUGGINGFACE_MODEL}  ({DENSE_DIM}-dim dense + BM25 sparse)")
    Milvus.from_documents(
        documents=documents,
        ids=ids,
        embedding=dense_ef,
        builtin_function=BM25BuiltInFunction(),
        vector_field=["dense", "sparse"],
        connection_args={"uri": f"http://{MILVUS_HOST}:{MILVUS_PORT}", "alias": "default"},
        collection_name=collection_name,
        drop_old=True,
    )
    print(f"\n[milvus] Ingestion complete -- {len(documents)} documents stored.")



# ---------------------------------------------------------------------------
# Test retrieval
# ---------------------------------------------------------------------------

def test_collection(
    collection_name: str,
    log_file: str,
) -> None:
    """
    Log every stored chunk with token counts, then run test queries.
    Results are written to log_file and summarised on stdout.
    Uses langchain_milvus to connect and retrieve all documents.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        filename=log_file, filemode="w", force=True,
    )
    log = logging.getLogger(__name__)

    # Connect using langchain_milvus directly
    vs = Milvus(
        embedding_function=dense_ef,
        builtin_function=BM25BuiltInFunction(),
        vector_field=["dense", "sparse"],
        connection_args={"uri": f"http://{MILVUS_HOST}:{MILVUS_PORT}", "alias": "default"},
        collection_name=collection_name,
    )
    
    # Retrieve all documents from the collection using a query iterator
    # (avoids the query() row cap and loading everything into memory at once)
    vs.col.load()
    pk_field = vs.col.schema.primary_field.name
    iterator = vs.col.query_iterator(
        expr=f'{pk_field} != ""',
        output_fields=["*"],
        batch_size=1000,
    )
    log.info(f"Collection : {collection_name}")

    doc_count = 0
    while True:
        batch = iterator.next()
        if not batch:
            iterator.close()
            break
        for result in batch:
            doc_count += 1
            text_content = result.get("text", "")
            token_count = len(tokenizer.encode(text_content))
            if token_count > MAX_TOKEN_LENGTH:
                print(f"WARNING: '{result.get('name','?')}' exceeds max token length "
                      f"({token_count} > {MAX_TOKEN_LENGTH}).")
            log.info(f"name   : {result.get('name') or result.get('source', 'unknown')}")
            log.info(f"tokens : {token_count}")
            log.info(f"source : {result.get('source', '')}")
            log.info(f"parent : {result.get('parent', '')}")
            log.info(f"ichunk : {result.get('ichunk', '')}")
            log.info(text_content)
            log.info("***\n")

    log.info(f"Total documents: {doc_count}")

    print("\n" + "=" * 72)
    print(f"\n[done] Full log written to: {log_file}")