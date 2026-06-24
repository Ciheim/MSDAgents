"""
Ingest FMSCoupler READMEs into Milvus.
Requires Milvus standalone on localhost:19530.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_utils import (
    chunkers, splitters, dense_ef, tokenizer,
    make_id, create_milvus_database)

from shared.metadata import ChunkMetadata

# ---------------------------------------------------------------------------
# Collection 
# ---------------------------------------------------------------------------

COLLECTION_NAME = "FMSCouplerDocs"
LOG_FILE  = f"{COLLECTION_NAME}.log"

# ---------------------------------------------------------------------------
# Documentation files
# ---------------------------------------------------------------------------

DOC_FILES = [
    "README.md",
    "FLUX.md",
    "AtmosDataType.md",
    "IceDataType.md",
    "LandDataType.md",
    "OceanPublicType.md",
    "OceanStateType.md",
    "AtmosIceBoundaryType.md",
    "AtmosLandBoundaryType.md",
    "IceOceanBoundaryType.md",
    "LandIceAtmosBoundaryType.md",
    "OceanIceBoundaryType.md",
    "IceOceanDriverType.md",
]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_doc(filepath: Path) -> tuple[list[Document], list[str]]:
    """Parse one *.md narrative doc into Documents with IDs."""

    documents: list[Document] = []
    ids: list[str] = []

    for section in splitters["readme"].split_text(filepath.read_text(encoding="utf-8")):
        content = section.page_content.strip()        
        parent = section.metadata.get("h1")
        name = make_id([section.metadata.get(h) for h in ("h1", "h2")])
        if section.metadata.get("h3"):
            name = make_id([section.metadata.get(h) for h in ("h1", "h2", "h3")])
            parent = make_id([parent, section.metadata.get("h2")])            
        splitted_content = chunkers["misc"].split_text(content)
        add_chunk = False if len(splitted_content) == 1 else True
        for ichunk, chunk_text in enumerate(splitted_content, start=1):
            name = f"{name}/chunk{ichunk}" if add_chunk else f"{name}"
            metadata = ChunkMetadata(
                source=filepath.name,
                name=name,
                parent=parent,
                datatype="readme",
                ichunk=0 if not add_chunk else ichunk,
            )
            documents.append(Document(page_content=chunk_text.strip(), metadata=metadata.model_dump()))
            ids.append(metadata.name)

    return documents, ids

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(docs_dir: Path|str, create_database: bool = False) -> tuple[list[Document], list[str]] | None:
    """Parse all documentation files."""
    all_documents: list[Document] = []
    all_ids: list[str] = []

    filepaths = [Path(docs_dir) / filename for filename in DOC_FILES]
    for filepath in filepaths:
        if not filepath.exists():
            raise FileNotFoundError(f"  {filepath.name}  MISSING")

    print(f"parse {len(DOC_FILES)}\n")
    for filepath in filepaths:
        docs, ids = parse_doc(filepath)
        all_documents.extend(docs)
        all_ids.extend(ids)
        print(f"  {filepath.name}  {len(docs)} documents")

    print(f"\n[chunk] Total: {len(all_documents)} documents")
    
    if create_database:
        create_milvus_database(all_documents, all_ids, COLLECTION_NAME)
    else:
        return all_documents, all_ids

# ---------------------------------------------------------------------------
# Test questions
# ---------------------------------------------------------------------------

TESTS = [
    ("What are the model component state types in the full coupler?",      None),
    ("How does the fast loop implicit tridiagonal diffusion scheme work?", None),
    ("What are the fields in atmos_data_type for radiative fluxes?",      "datatype"),
    ("What fields does ice_ocean_boundary_type pass from ice to ocean?",  "boundary_type"),
    ("How is MPI PE layout configured for atmosphere and ocean?",          "overview"),
    ("What controls concurrent radiation in OpenMP threading?",            "overview"),
    ("How is the model start time determined from coupler.res?",           None),
    ("What is the difference between fast ice and slow ice physics?",      None),
    ("What are the REGRID, REDIST, and DIRECT flux transfer modes?",      "flux_exchange"),
    ("What does the ice_ocean_driver_type control structure do?",         "driver_type"),
]

