from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from pymilvus import DataType, MilvusClient

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PARSER_MODULE_PATH = WORKSPACE_ROOT / "parsers" / "fortran_parser" / "doxygen_xml_parser.py"

parser_spec = importlib.util.spec_from_file_location("fms_doxygen_xml_parser", PARSER_MODULE_PATH)
if parser_spec is None or parser_spec.loader is None:
    raise ImportError(f"Cannot load parser module from {PARSER_MODULE_PATH}")
parser_module = importlib.util.module_from_spec(parser_spec)
parser_spec.loader.exec_module(parser_module)
ModuleBodyDocument = parser_module.ModuleBodyDocument


def parse_xml_directory(xml_dir: Path, markdown_dir: Path) -> list[Document]:
    """Parse Doxygen module XML files into LangChain documents."""
    docs: list[Document] = []

    xml_files = sorted(Path(xml_dir).glob("*.xml"))
    if not xml_files:
        return docs

    for xml_file in xml_files:
        xml_name = xml_file.name
        if not xml_name.startswith("namespace"):
            continue
        if not xml_name.endswith("__mod.xml"):
            continue

        try:
            parsed = _parse_single_module(xml_dir=xml_dir, xml_name=xml_name, markdown_dir=markdown_dir)
        except Exception as exc:
            print(f"Skipping {xml_name}: {exc}")
            continue
        docs.extend(parsed)

    return docs


def _parse_single_module(xml_dir: Path, xml_name: str, markdown_dir: Path) -> list[Document]:
    """Parse one module XML file and return procedure + variable documents."""
    try:
        module_doc = ModuleBodyDocument(
            xmldir=xml_dir,
            xmlfile=xml_name,
            append_overview=True,
            include_flowchart=False,
        )
    except Exception:
        # Some module files may not have a matching top-level overview XML.
        module_doc = ModuleBodyDocument(
            xmldir=xml_dir,
            xmlfile=xml_name,
            append_overview=False,
            include_flowchart=False,
        )

    module_doc.document_module_variables()
    module_doc.document_procedures()

    markdown_dir.mkdir(parents=True, exist_ok=True)
    original_cwd = Path.cwd()
    try:
        # The parser writes markdown to the current working directory.
        os.chdir(markdown_dir)
        markdown_file = module_doc.write_markdown()
    finally:
        os.chdir(original_cwd)

    out_docs: list[Document] = []
    module_name = str(module_doc.toplevel_name)
    
    # Create documents for each procedure using individual procedure names
    for i, procedure_md in enumerate(module_doc.procedures_md):
        procedure_name = module_doc.procedure_names[i] if i < len(module_doc.procedure_names) else f"procedure_{i}"
        out_docs.append(
            Document(
                page_content=procedure_md,
                metadata={
                    "source": module_name,
                    "name": procedure_name,  # Use individual procedure name, not module name
                    "kind": "procedure",
                    "xml_file": xml_name,
                    "markdown_file": markdown_file,
                },
            )
        )

    # Create separate documents for each variable using individual variable names
    if module_doc.variables_md and module_doc.variable_names:
        # Skip the header and table format lines (first 2 items)
        variable_rows = module_doc.variables_md[2:-1]  # Exclude header, format line, and trailing newline
        
        for i, var_row in enumerate(variable_rows):
            if i < len(module_doc.variable_names):
                variable_name = module_doc.variable_names[i]
                out_docs.append(
                    Document(
                        page_content=var_row,
                        metadata={
                            "source": module_name,
                            "name": variable_name,  # Use individual variable name, not module name
                            "kind": "variable",
                            "xml_file": xml_name,
                            "markdown_file": markdown_file,
                        },
                    )
                )

    return out_docs


def ensure_collection(client: MilvusClient, collection_name: str, recreate: bool) -> None:
    """Create collection if missing, or recreate if requested."""
    exists = client.has_collection(collection_name=collection_name)
    if exists and recreate:
        client.drop_collection(collection_name=collection_name)
        exists = False

    if not exists:
        schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="name", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="kind", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="xml_file", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="markdown_file", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=2)

        client.create_collection(collection_name=collection_name, schema=schema)


def batch_chunks(items: list[Document], chunk_size: int) -> Iterable[list[Document]]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def ingest_documents(
    client: MilvusClient,
    collection_name: str,
    documents: list[Document],
    batch_size: int,
) -> int:
    """Insert parsed documents into Milvus."""
    inserted = 0
    default_embedding = [0.0, 0.0]

    for chunk in batch_chunks(documents, batch_size):
        rows: list[dict[str, object]] = []
        for doc in chunk:
            metadata = dict(doc.metadata)
            rows.append(
                {
                    "text": doc.page_content,
                    "source": str(metadata.get("source", "")),
                    "name": str(metadata.get("name", "")),
                    "kind": str(metadata.get("kind", "")),
                    "xml_file": str(metadata.get("xml_file", "")),
                    "markdown_file": str(metadata.get("markdown_file", "")),
                    "embedding": default_embedding,
                }
            )

        client.insert(collection_name=collection_name, data=rows)

        inserted += len(chunk)

    return inserted


def connect_client(milvus_db_path: Path) -> MilvusClient:
    """Connect to a local Milvus Lite database file."""
    milvus_db_path.parent.mkdir(parents=True, exist_ok=True)
    return MilvusClient(uri=str(milvus_db_path))


def main() -> int:
    milvus_db_path = Path("/home/Ryan.Mulhall/msdagents/fms-chatbot/local_storage/fms_milvus.db")
    collection_name = "fms"
    xml_dir = "/home/Ryan.Mulhall/msdagents/fms/build_docs/docs/xml"
    markdown_export_dir = Path("/home/Ryan.Mulhall/msdagents/fms-chatbot/local_storage/parsed_modules")
    batch_size = 100
    recreate_collection = True

    if not Path(xml_dir).exists() or not Path(xml_dir).is_dir():
        Path(xml_dir).mkdir(parents=True, exist_ok=True)
    if batch_size <= 0:
        print("batch-size must be > 0")
        return 1

    docs = parse_xml_directory(xml_dir=Path(xml_dir), markdown_dir=markdown_export_dir)
    if not docs:
        print("No parseable module XML files found (expected namespace*__mod.xml files).")
        return 1

    client = connect_client(milvus_db_path)
    try:
        ensure_collection(client, collection_name, recreate_collection)
        inserted = ingest_documents(client, collection_name, docs, batch_size)
    finally:
        if hasattr(client, "close"):
            client.close()

    print(f"Parsed documents: {len(docs)}")
    print(f"Markdown export dir: {markdown_export_dir}")
    print(f"Inserted documents: {inserted}")
    print(f"Collection: {collection_name}")
    print(f"Milvus DB: {milvus_db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
