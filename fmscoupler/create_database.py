"""
create_database.py

Unified database builder: combines narrative documentation and Doxygen-generated
code documentation into a single FMSCoupler collection.

  FMSCouplerDocs  — narrative documentation  (create_readmes.py)
  FMSCouplerCode  — Doxygen module/subroutine docs (document_code.py)
  FMSCoupler      — unified collection combining both sources

Usage
-----
    python create_database.py   # build unified collection
"""

from pathlib import Path
import os
import shutil
from collections import Counter

import document_readmes
import document_code
from document_utils import create_milvus_database, test_collection

COLLECTION_NAME = "FMSCoupler"
LOG_FILE = "create_database.log"
DOCS_DIR = Path("fmscoupler/full/docs")
CODE_MODS_DIR = Path("fmscoupler/docs")

TESTS = [
    # Documentation tests
    ("What are the model component state types in the full coupler?",      None),
    ("How does the fast loop implicit tridiagonal diffusion scheme work?", None),
    ("What are the fields in atmos_data_type for radiative fluxes?",      "datatype"),
    ("What fields does ice_ocean_boundary_type pass from ice to ocean?",  "boundary_type"),
    ("How is MPI PE layout configured for atmosphere and ocean?",          "overview"),
    
    # Code tests
    ("What does coupler_init do and what arguments does it take?",       None),
    ("How does sfc_boundary_layer work step by step?",                   None),
    ("What module variables are defined in full_coupler_mod?",           None),
    ("What are the steps in the atmos_ocean_fluxes_calc flowchart?",     None),
    ("How does land_ice_flux_exchange compute turbulent fluxes?",        None),
]


def clone_repository_and_run_doxygen(repository_path: str, repository_name: str, branch: str) -> Path:
    """Clone a repository branch and run doxygen in the cloned directory."""
    
    print(f"Cloning {repository_name} from {repository_path} on branch {branch}...")
    clone_rc = os.system(f"git clone -b {branch} {repository_path}/{repository_name}.git")
    if clone_rc != 0:
        raise RuntimeError(
            f"Failed to clone '{repository_name}' on branch '{branch}'."
        )

    # hack
    repo_dir = Path(repository_name)
    for dirname in ("simple", "shared", "SHiELD"):
        target_dir = repo_dir / dirname
        if target_dir.exists() and target_dir.is_dir():
            shutil.rmtree(target_dir)
            print(f"Removed directory: {target_dir}")

    print(f"Running doxygen in {repository_name}...")
    doxygen_rc = os.system(f"cd {repository_name} && doxygen")
    if doxygen_rc != 0:
        raise RuntimeError(f"doxygen command failed")

    return repository_name

if __name__ == "__main__":
    print("=" * 72)
    print("Building unified FMSCoupler collection")
    print("=" * 72)
    
    clone_repository_and_run_doxygen("https://github.com/mlee03", "fmscoupler", "doc/all-round1")

    # Get documents from both sources without creating separate databases
    print("parse Gathering narrative documentation...")
    readme_docs, readme_ids = document_readmes.build(docs_dir=DOCS_DIR, create_database=False)
    print(f"  -> {len(readme_docs)} documents from narrative docs")
    
    print("parse Gathering code module documentation...")
    code_docs, code_ids = document_code.build(code_mods_dir=CODE_MODS_DIR, create_database=False)
    print(f"  -> {len(code_docs)} documents from code modules")
    
    # Combine all documents and ids
    all_documents = readme_docs + code_docs
    all_ids = readme_ids + code_ids

    print("checking duplicate document IDs...")
    uniqueids = set(all_ids)
    if len(uniqueids) != len(all_ids):
        duplicates = [item for item, count in Counter(all_ids).items() if count > 1]        
        raise RuntimeError(f"Duplicate document IDs found: {duplicates}")
    
    print(f"\n[chunk] Total combined: {len(all_documents)} documents")
    print(f"[build] Creating unified Milvus collection: {COLLECTION_NAME}")
    
    # Create the unified database
    create_milvus_database(all_documents, all_ids, COLLECTION_NAME)
    
    # Test the unified collection
    print(f"\n[test] Testing unified collection...")
    test_collection(COLLECTION_NAME, TESTS, LOG_FILE)
    
    print(f"\nDone. Unified collection available:")
    print(f"  {COLLECTION_NAME:<24}  log: {LOG_FILE}")

