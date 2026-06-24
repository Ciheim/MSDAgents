from __future__ import annotations

import logging

# Suppress gRPC debug logs (too_many_pings warnings)
logging.getLogger("grpc").setLevel(logging.WARNING)

import re
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


LLM_MODEL = "llama3.2"
MILVUS_DB_PATH = Path("/home/Ryan.Mulhall/msdagents/fms-chatbot/local_storage/fms_milvus.db")
COLLECTION_NAME = "fms"
TOP_K = 15
MAX_CANDIDATES = 2000


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


def fetch_context(client: MilvusClient, query: str, limit: int = TOP_K) -> str:
  """Retrieve matching docs from Milvus and format them as context."""
  # Pull a bounded candidate set from Milvus, then rank by simple keyword overlap.
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
  except Exception as exc:
    return f"Failed to query Milvus collection '{COLLECTION_NAME}': {exc}"

  scored_rows = [(_keyword_score(query, row), row) for row in candidates]
  scored_rows.sort(key=lambda item: item[0], reverse=True)
  top_rows = [row for score, row in scored_rows if score > 0][:limit]
  if not top_rows:
    top_rows = [row for _, row in scored_rows[:limit]]

  chunks: list[str] = []
  for idx, props in enumerate(top_rows, start=1):
    chunks.append(
      (
        f"Context {idx}:\n"
        f"name: {props.get('name', '')}\n"
        f"kind: {props.get('kind', '')}\n"
        f"source: {props.get('source', '')}\n"
        f"xml_file: {props.get('xml_file', '')}\n"
        f"markdown_file: {props.get('markdown_file', '')}\n"
        f"text: {props.get('text', '')}"
      )
    )

  if not chunks:
    return "No relevant context was found in the FMS Milvus collection."

  return "\n\n".join(chunks)


def main() -> None:
  chatbot = ChatOllama(model=LLM_MODEL, microstat_tau=2.0)

  MILVUS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
  client = MilvusClient(uri=str(MILVUS_DB_PATH))
  try:
    if not client.has_collection(collection_name=COLLECTION_NAME):
      raise RuntimeError(
        f"Collection '{COLLECTION_NAME}' not found. Run create_fms_database.py first."
      )

    system_message = SystemMessage(
      content=(
        "FMS is the Flexible Modeling System, a Fortran library used for scientific computing in climate simulations."
        "You are an FMS coding assistant to answer questions about FMS routines and modules. "
        "Only answer questions using the retrieved context. "
        "If context is insufficient, say you do not have enough information "
        "from the indexed FMS docs."
        "Ensure that any code examples you provide are valid Fortran code. "
        "FMS contains many interfaces to provide generic interfaces to different data types, "
        "which should be used instead of calling their routines directly. "
        "If a routine belongs to a generic interface, provide the name of the generic interface in your answer first. "
      )
    )

    intro_query = "Introduce yourself. Ask how may I assist you?"
    intro_context = fetch_context(client, intro_query)
    intro = chatbot.invoke(
      [
        system_message,
        HumanMessage(
          content=(
            f"Question: {intro_query}\n\n"
            f"Retrieved context:\n{intro_context}\n\n"
            "Answer using only the retrieved context."
          )
        ),
      ]
    )

    print(intro.content)
    print(">", end=" ")

    while True:
      query = input().strip()
      if not query:
        print(">", end=" ")
        continue
      if "goodbye" in query.lower() or query.lower() in {"exit", "quit"}:
        print("Goodbye.")
        break

      context = fetch_context(client, query)
      answer = chatbot.invoke(
        [
          system_message,
          HumanMessage(
            content=(
              f"Question: {query}\n\n"
              f"Retrieved context:\n{context}\n\n"
              "Answer using only the retrieved context."
            )
          ),
        ]
      )
      print(answer.content)
      print(">", end=" ")
  finally:
    if hasattr(client, "close"):
      client.close()


if __name__ == "__main__":
  main()

