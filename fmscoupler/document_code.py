"""
document_code.py

Ingest FMSCoupler Doxygen-generated module documentation into Milvus.
Collection : FMSCouplerCode

Usage
-----
    python document_code.py

Requires Milvus standalone on localhost:19530.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_utils import (
    dense_ef, tokenizer, MAX_TOKEN_LENGTH, CHUNK_OVERLAP,
    h12_splitter, h3_splitter, make_id, create_milvus_database,
    HUGGINGFACE_MODEL, MILVUS_HOST, MILVUS_PORT,
)

from shared.metadata import ChunkMetadata
from parsers.fortran_parser import doxygen_xml_parser

# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

COLLECTION_NAME = "FMSCouplerCode"
LOG_FILE = f"{COLLECTION_NAME}.log"

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------

XMLFILES = {
    "atm_land_ice_flux_exchange_mod": "namespaceatm__land__ice__flux__exchange__mod.xml",
    "atmos_ocean_dep_fluxes_calc_mod": "namespaceatmos__ocean__dep__fluxes__calc__mod.xml",
    "atmos_ocean_fluxes_calc_mod": "namespaceatmos__ocean__fluxes__calc__mod.xml",
    "flux_exchange_mod": "namespaceflux__exchange__mod.xml",
    "full_coupler_mod": "namespacefull__coupler__mod.xml",
    "ice_ocean_flux_exchange_mod": "namespaceice__ocean__flux__exchange__mod.xml",
    "land_ice_flux_exchange_mod": "namespaceland__ice__flux__exchange__mod.xml",
}

# Text splitters keyed by h3 subsection type (used in add_chunk)
SUBSECTION_SPLITTERS = {
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
    )
}

def xml_to_markdown() -> list[str]:
    """
    Convert Doxygen-generated XML files to Markdown files.
    """
    readmes = []

    #hack
    if Path("fmscoupler/docs/xml/full_2flux__exchange_8_f90.xml").exists():
        print("Renaming flux__exchange_8_f90.xml to full_2flux__exchange_8_f90.xml...")
        Path("fmscoupler/docs/xml/full_2flux__exchange_8_f90.xml").rename("fmscoupler/docs/xml/flux__exchange_8_f90.xml")

    for module, xmlfile in XMLFILES.items():
        print(f"Processing {module}...")
        modxml = doxygen_xml_parser.ModuleBodyDocument(xmldir="fmscoupler/docs/xml", xmlfile=xmlfile)
        modxml.document_module_variables()
        modxml.document_procedures()
        readmes.append(Path(modxml.write_markdown(output_dir=".")))
    return readmes

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_doc(filepath: Path) -> tuple[list[Document], list[str]]:
    """
    Parse one Doxygen-generated code-module .md file.
    """

    documents: list[Document] = []
    ids: list[str] = []

    for section in h12_splitter.split_text(filepath.read_text(encoding="utf-8")):
        h1 = section.metadata.get("h1", "") #module name
        h2 = section.metadata.get("h2", "") #variable, subroutine::subroutine_name, function::function_name
        content = section.page_content.strip()

        section_id = "/".join(p for p in [h1, h2] if p)

        if "variable" in h2:
            # each variable is a document
            for ivar in content.splitlines():                
                name = make_id([h1, h2, ivar.strip()])
                metadata = ChunkMetadata(
                    source=h1, 
                    name=name,
                    parent=h1, 
                    datatype="variable"
                )
                documents.append(Document(page_content=ivar.strip(), metadata=metadata.model_dump()))
                ids.append(name)

        elif "subroutine" in h2 or "function" in h2:
            # Procedure block — h3 section is either "flowchart", "arguments", "intro", or "description"
            for subsection in h3_splitter.split_text(content):
                h3 = subsection.metadata.get("h3")
                splitter = SUBSECTION_SPLITTERS.get(h3, SUBSECTION_SPLITTERS["description"])
                splitted_content = splitter.split_text(subsection.page_content)
                for ichunk, chunk in enumerate(splitted_content, start=1):
                    name = make_id([h1, h2, h3, f"chunk{ichunk}"])
                    metadata = ChunkMetadata(
                        source=h1,
                        name=name,
                        parent=section_id,
                        datatype="procedure",
                        ichunk=ichunk
                    )
                    documents.append(Document(page_content=chunk.strip(), metadata=metadata.model_dump()))
                    ids.append(name)

    return documents, ids

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(code_mods_dir: Path|str, create_database: bool = False) -> tuple[list[Document], list[str]] | None:
    """Parse all code-module .md files."""

    filepaths = xml_to_markdown()  # Convert XML to Markdown files
    for filepath in filepaths:
        if not filepath.exists():
            raise FileNotFoundError(f"  {filepath.name}  MISSING")

    all_documents: list[Document] = []
    all_ids: list[str] = []

    for filepath in filepaths:
        docs, ids = parse_doc(filepath)
        all_documents.extend(docs)
        all_ids.extend(ids)
        print(f"  {filepath.name:<45}  {len(docs):>3} documents")

    print(f"\nTotal: {len(all_documents)} documents")
    
    if create_database:
        create_milvus_database(all_documents, all_ids, COLLECTION_NAME)
    else:
        return all_documents, all_ids

# ---------------------------------------------------------------------------
# Test questions
# ---------------------------------------------------------------------------

TESTS = [
    ("What does coupler_init do and what arguments does it take?",       None),
    ("What arguments does flux_exchange_init accept?",                   None),
    ("How does sfc_boundary_layer work step by step?",                   None),
    ("What module variables are defined in full_coupler_mod?",           None),
    ("What are the steps in the atmos_ocean_fluxes_calc flowchart?",     None),
    ("How does land_ice_flux_exchange compute turbulent fluxes?",        None),
    ("What subroutines does atm_land_ice_flux_exchange_mod provide?",    None),
    ("How are ice-ocean fluxes calculated in ice_ocean_flux_exchange?",  None),
]

