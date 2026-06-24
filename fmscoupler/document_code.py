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

from document_utils import (
    chunkers, splitters, dense_ef, tokenizer, MAX_TOKEN_LENGTH, CHUNK_OVERLAP,
    make_id, create_milvus_database,
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

    for section in splitters["module"].split_text(filepath.read_text(encoding="utf-8")):
        source = section.metadata.get("h1", "") #module name
        h2 = section.metadata.get("h2", "") #variable, subroutine_name, or function_name
        content = section.page_content.strip()

        if "variable" in h2:
            # each variable is a document
            for ivar in content.splitlines():
                name = make_id([source, ivar.strip()])                
                metadata = ChunkMetadata(
                    source=source, 
                    name=name,
                    parent=source, 
                    datatype="variable"
                )
                documents.append(Document(page_content=ivar.strip(), metadata=metadata.model_dump()))
                ids.append(name)

        elif "subroutine" in h2 or "function" in h2:
            # h3 section is either "flowchart", "arguments", "intro", or "description"
            for subsection in splitters["subroutine"].split_text(content):
                subsectiontype = subsection.metadata.get("h3")
                splitted_content = chunkers.get(subsectiontype).split_text(subsection.page_content)
                add_chunk = False if len(splitted_content) == 1 else True
                name = make_id([source, h2, subsectiontype])
                for ichunk, chunk in enumerate(splitted_content, start=1):
                    name = make_id([name, f"chunk{ichunk}"]) if add_chunk else name
                    metadata = ChunkMetadata(
                        source=source,
                        name=name,
                        parent=make_id([source, h2]),
                        datatype="procedure",
                        ichunk=0 if not add_chunk else ichunk
                    )
                    documents.append(Document(page_content=chunk.strip(), metadata=metadata.model_dump()))
                    ids.append(name)

    return documents, ids

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(code_mods_dir: Path|str, create_database: bool = False, cleanup: bool = True) -> tuple[list[Document], list[str]] | None:
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
    
    if cleanup:
        for filepath in filepaths:
            filepath.unlink()  # Remove the generated Markdown files
    
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

