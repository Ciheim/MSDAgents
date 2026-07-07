from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import DataType, MilvusClient

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PARSER_MODULE_PATH = WORKSPACE_ROOT / "parsers" / "fortran_parser" / "doxygen_xml_parser.py"

parser_spec = importlib.util.spec_from_file_location("fms_doxygen_xml_parser", PARSER_MODULE_PATH)
if parser_spec is None or parser_spec.loader is None:
    raise ImportError(f"Cannot load parser module from {PARSER_MODULE_PATH}")
parser_module = importlib.util.module_from_spec(parser_spec)
parser_spec.loader.exec_module(parser_module)
ModuleBodyDocument = parser_module.ModuleBodyDocument
InterfaceDocument = parser_module.InterfaceDocument

# TODO consider using a GPU-enabled model for embeddings, hits cuda version mismatches on amd
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def parse_xml_directory(
    xml_dir: Path,
    markdown_dir: Path,
    include_flowchart: bool,
) -> list[Document]:
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
            parsed = _parse_single_module(
                xml_dir=xml_dir,
                xml_name=xml_name,
                markdown_dir=markdown_dir,
                include_flowchart=include_flowchart,
            )
        except Exception as exc:
            print(f"Skipping {xml_name}: {exc}")
            continue
        docs.extend(parsed)

    for xml_file in xml_files:
        xml_name = xml_file.name
        if not xml_name.startswith("interface"):
            continue

        try:
            parsed = _parse_single_interface(
                xml_dir=xml_dir,
                xml_name=xml_name,
                markdown_dir=markdown_dir,
                include_flowchart=include_flowchart,
            )
        except Exception as exc:
            print(f"Skipping {xml_name}: {exc}")
            continue
        docs.extend(parsed)

    return docs


def _parse_single_module(
    xml_dir: Path,
    xml_name: str,
    markdown_dir: Path,
    include_flowchart: bool,
) -> list[Document]:
    """Parse one module XML file and return procedure + variable documents."""
    try:
        module_doc = ModuleBodyDocument(
            xmldir=xml_dir,
            xmlfile=xml_name,
            append_overview=True,
            include_flowchart=include_flowchart,
        )
    except Exception:
        # Some module files may not have a matching top-level overview XML.
        module_doc = ModuleBodyDocument(
            xmldir=xml_dir,
            xmlfile=xml_name,
            append_overview=False,
            include_flowchart=include_flowchart,
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

    procedure_names = [
        module_doc.get_name(proc)
        for proc in module_doc.soup.find_all("memberdef", {"kind": "function"})
    ]
    variable_names = [
        module_doc.get_name(var)
        for var in module_doc.soup.find_all("memberdef", {"kind": "variable"})
    ]
    
    # Create documents for each procedure using individual procedure names
    for i, procedure_md in enumerate(module_doc.procedures_md):
        procedure_name = procedure_names[i] if i < len(procedure_names) else f"procedure_{i}"
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
    if module_doc.variables_md and variable_names:
        # Skip the header and table format lines (first 2 items)
        variable_rows = module_doc.variables_md[2:-1]  # Exclude header, format line, and trailing newline
        
        for i, var_row in enumerate(variable_rows):
            if i < len(variable_names):
                variable_name = variable_names[i]
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


def _parse_single_interface(
    xml_dir: Path,
    xml_name: str,
    markdown_dir: Path,
    include_flowchart: bool,
) -> list[Document]:
    """Parse one interface XML file and associate it with its owning module."""
    interface_doc = InterfaceDocument(
        xmldir=xml_dir,
        xmlfile=xml_name,
        include_flowchart=include_flowchart,
    )
    interface_markdown = interface_doc.document_interface()

    markdown_dir.mkdir(parents=True, exist_ok=True)
    original_cwd = Path.cwd()
    try:
        os.chdir(markdown_dir)
        markdown_file = interface_doc.write_markdown()
    finally:
        os.chdir(original_cwd)

    metadata = {
        "source": interface_doc.module_name or interface_doc.interface_name,
        "name": interface_doc.generic_name,
        "kind": "interface",
        "xml_file": xml_name,
        "markdown_file": markdown_file,
    }

    docs = [
        Document(
            page_content=interface_markdown,
            metadata=metadata,
        )
    ]

    for procedure in interface_doc.soup.find_all("memberdef", {"kind": "function"}):
        procedure_name = interface_doc.get_name(procedure)
        procedure_type = interface_doc.get_tag_to_string("type", procedure).split(",")[0].strip()
        parameters_description = interface_doc.get_parameters_description(
            procedure,
            subroutine_name=procedure_name,
        )
        briefdescription = interface_doc.get_tag_to_string("briefdescription", procedure)
        detaileddescription = interface_doc.get_tag_to_string("detaileddescription", procedure)

        if briefdescription and briefdescription[-1] != ".":
            briefdescription += "."
        if detaileddescription and detaileddescription[-1] != ".":
            detaileddescription += "."

        procedure_markdown = (
            f"## {procedure_name}\n"
            f"### intro\n"
            f"{procedure_name} is a {procedure_type} implementation of the "
            f"{interface_doc.generic_name} interface in {metadata['source']}.\n"
            f"### description\n"
            f"{briefdescription}  {detaileddescription}\n"
            f"### arguments\n{parameters_description}\n"
        )

        docs.append(
            Document(
                page_content=procedure_markdown,
                metadata={
                    **metadata,
                    "name": procedure_name,
                    "interface_name": interface_doc.generic_name,
                    "kind": "interface_procedure",
                },
            )
        )

    return docs


def ensure_collection(
    client: MilvusClient,
    collection_name: str,
    recreate: bool,
    vector_dim: int,
) -> None:
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
        schema.add_field(field_name="interface_name", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=vector_dim)

        client.create_collection(collection_name=collection_name, schema=schema)


def batch_chunks(items: list[Document], chunk_size: int) -> Iterable[list[Document]]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def ingest_documents(
    client: MilvusClient,
    collection_name: str,
    documents: list[Document],
    batch_size: int,
    embeddings: HuggingFaceEmbeddings,
) -> int:
    """Insert parsed documents into Milvus."""
    inserted = 0

    for chunk in batch_chunks(documents, batch_size):
        chunk_texts = [doc.page_content for doc in chunk]
        chunk_embeddings = embeddings.embed_documents(chunk_texts)

        rows: list[dict[str, object]] = []
        for doc, embedding in zip(chunk, chunk_embeddings):
            metadata = dict(doc.metadata)
            rows.append(
                {
                    "text": doc.page_content,
                    "source": str(metadata.get("source", "")),
                    "name": str(metadata.get("name", "")),
                    "kind": str(metadata.get("kind", "")),
                    "xml_file": str(metadata.get("xml_file", "")),
                    "markdown_file": str(metadata.get("markdown_file", "")),
                    "interface_name": str(metadata.get("interface_name", "")),
                    "embedding": embedding,
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
    include_flowchart = False

    if not Path(xml_dir).exists() or not Path(xml_dir).is_dir():
        Path(xml_dir).mkdir(parents=True, exist_ok=True)
    if batch_size <= 0:
        print("batch-size must be > 0")
        return 1

    docs = parse_xml_directory(
        xml_dir=Path(xml_dir),
        markdown_dir=markdown_export_dir,
        include_flowchart=include_flowchart,
    )

    if not docs:
        print("No parseable module XML files found (expected namespace*__mod.xml files).")
        return 1

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_dim = len(embeddings.embed_query("vector dimension probe"))

    client = connect_client(milvus_db_path)
    try:
        ensure_collection(client, collection_name, recreate_collection, vector_dim)
        inserted = ingest_documents(client, collection_name, docs, batch_size, embeddings)
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
