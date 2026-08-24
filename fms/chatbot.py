from __future__ import annotations

import logging

# Suppress gRPC debug logs (too_many_pings warnings)
logging.getLogger("grpc").setLevel(logging.WARNING)

import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import MilvusClient

from shared.chatbot import RAGChatbot


LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MILVUS_DB_PATH = Path("/home/Ryan.Mulhall/msdagents/fms-chatbot/local_storage/fms_milvus.db")
COLLECTION_NAME = "fms"
TOP_K = 15
MAX_CANDIDATES = 2000

SYSTEM_MESSAGE = (
  "FMS is the Flexible Modeling System, a Fortran library used for scientific computing in climate simulations. "
  "You are an FMS coding assistant to answer questions about FMS routines and modules. "
  "Only answer questions using the retrieved context. "
  "If context is insufficient, say you do not have enough information from the indexed FMS docs. "
  "Ensure that any code examples you provide are valid Fortran code. "
  "FMS contains many interfaces to provide generic interfaces to different data types, "
  "which should be used instead of calling their routines directly. "
  "If a routine belongs to a generic interface, provide the name of the generic interface in your answer first. "
  "\n\nContext:\n{context}"
)


def _tokenize(text: str) -> list[str]:
  return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _keyword_score(query: str, record: dict[str, Any]) -> int:
  tokens = [t for t in _tokenize(query) if len(t) > 1]
  if not tokens:
    return 0

  searchable = " ".join(
    [
      str(record.get("name", "")),
      str(record.get("source", "")),
      str(record.get("kind", "")),
      str(record.get("xml_file", "")),
      str(record.get("markdown_file", "")),
      str(record.get("text", "")),
    ]
  ).lower()
  return sum(searchable.count(token) for token in tokens)


def _record_to_document(props: dict[str, Any]) -> Document:
  return Document(
    page_content=(
      f"name: {props.get('name', '')}\n"
      f"kind: {props.get('kind', '')}\n"
      f"source: {props.get('source', '')}\n"
      f"xml_file: {props.get('xml_file', '')}\n"
      f"markdown_file: {props.get('markdown_file', '')}\n"
      f"text: {props.get('text', '')}"
    ),
    metadata={
      "name": str(props.get("name", "")),
      "kind": str(props.get("kind", "")),
      "source": str(props.get("source", "")),
      "xml_file": str(props.get("xml_file", "")),
      "markdown_file": str(props.get("markdown_file", "")),
    },
  )


def _keyword_retrieve(
  client: MilvusClient,
  query: str,
  limit: int,
) -> list[tuple[Document, float]]:
  try:
    candidates = client.query(
      collection_name=COLLECTION_NAME,
      output_fields=["text", "name", "source", "kind", "xml_file", "markdown_file"],
      limit=MAX_CANDIDATES,
    )
  except TypeError:
    candidates = client.query(
      collection_name=COLLECTION_NAME,
      filter="",
      output_fields=["text", "name", "source", "kind", "xml_file", "markdown_file"],
      limit=MAX_CANDIDATES,
    )

  scored_rows = [(_keyword_score(query, row), row) for row in candidates]
  scored_rows.sort(key=lambda item: item[0], reverse=True)
  selected_rows = [(score, row) for score, row in scored_rows if score > 0][:limit]
  if not selected_rows:
    selected_rows = scored_rows[:limit]

  return [(_record_to_document(props), float(score)) for score, props in selected_rows]


def retrieve_documents(
  client: MilvusClient,
  embeddings: HuggingFaceEmbeddings,
  query: str,
  limit: int = TOP_K,
) -> list[tuple[Document, float]]:
  """Retrieve matching docs from Milvus vector search with keyword fallback."""

  query_vector = embeddings.embed_query(query)
  search_results = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    anns_field="embedding",
    output_fields=["text", "name", "source", "kind", "xml_file", "markdown_file"],
    limit=limit,
  )

  first = search_results[0]
  hits = first

  docs_and_scores: list[tuple[Document, float]] = []
  for hit in hits:
    props: dict[str, Any] = {}
    distance = 0.0

    if isinstance(hit, dict):
      entity = hit.get("entity")
      if isinstance(entity, dict):
        props = entity
      else:
        props = {
          "text": hit.get("text", ""),
          "name": hit.get("name", ""),
          "source": hit.get("source", ""),
          "kind": hit.get("kind", ""),
          "xml_file": hit.get("xml_file", ""),
          "markdown_file": hit.get("markdown_file", ""),
        }
      distance = float(hit.get("distance", hit.get("score", 0.0)) or 0.0)

    if not props:
      continue

    similarity = distance #1.0 / (1.0 + max(distance, 0.0))
    docs_and_scores.append((_record_to_document(props), similarity))

  if docs_and_scores:
    return docs_and_scores

  try:
    docs_and_scores = _keyword_retrieve(client, query, limit)
    if docs_and_scores:
      return docs_and_scores
  except Exception:
    pass

  fallback_doc = Document(
    page_content="No relevant context was found in the FMS Milvus collection.",
    metadata={"source": "milvus", "name": "empty_result", "kind": "info"},
  )
  return [(fallback_doc, 0.0)]


def main() -> None:
  MILVUS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
  client = MilvusClient(uri=str(MILVUS_DB_PATH))
  try:
    if not client.has_collection(collection_name=COLLECTION_NAME):
      raise RuntimeError(
        f"Collection '{COLLECTION_NAME}' not found. Run create_fms_database.py first."
      )

    embeddings = HuggingFaceEmbeddings(
      model_name=EMBEDDING_MODEL,
      model_kwargs={"device": "cpu"},
      encode_kwargs={"normalize_embeddings": True},
    )

    chatbot = RAGChatbot(
      vectorstore=None,
      system_message=SYSTEM_MESSAGE,
      model_name=LLM_MODEL,
      retrieve_function=lambda question: retrieve_documents(client, embeddings, question),
    )


    print("FMS assistant ready. Ask a question (type 'exit' to quit).")
    print(">", end=" ")

    while True:
      query = input().strip()
      if not query:
        print(">", end=" ")
        continue
      if "goodbye" in query.lower() or query.lower() in {"exit", "quit"}:
        print("Goodbye.")
        break

      answer, docs_and_scores, _ = chatbot.ask(query)
      print(answer)

      sources = ", ".join(
        [
          f"{doc.metadata.get('source', '')}/{doc.metadata.get('name', '')}"
          for doc, _ in docs_and_scores
        ]
      )
      if sources:
        print(f"Sources: {sources}")
      print(">", end=" ")
  finally:
    if hasattr(client, "close"):
      client.close()


if __name__ == "__main__":
  main()

