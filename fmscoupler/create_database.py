"""
Creates FMSCoupler database

Usage:
python create_database.py  
"""

from pathlib import Path
import subprocess
import shutil
from collections import Counter

import document_readmes
import document_code
from document_utils import create_milvus_database, test_collection

COLLECTION_NAME = "FMSCoupler"
LOG_FILE = "create_database.log"
DOCS_DIR = Path("fmscoupler/full/docs")
CODE_MODS_DIR = Path("fmscoupler/docs")


def clone_repository_and_run_doxygen(repository_path: str, repository_name: str, branch: str) -> Path:
    """Clone a repository branch and run doxygen in the cloned directory."""
    
    repo_dir = Path(repository_name)
    if repo_dir.exists():
        print(f"Removing existing directory: {repo_dir}")
        shutil.rmtree(repo_dir)

    print(f"Cloning {repository_name} from {repository_path} on branch {branch}...")
    clone_result = subprocess.run(
        ["git", "clone", "-b", branch, f"{repository_path}/{repository_name}.git"]
    )
    if clone_result.returncode != 0:
        raise RuntimeError(
            f"Failed to clone '{repository_name}' on branch '{branch}'."
        )

    # hack
    for dirname in ("simple", "shared", "SHiELD"):
        target_dir = repo_dir / dirname
        if target_dir.exists():
            shutil.rmtree(target_dir)
            print(f"Removed directory: {target_dir}")

    print(f"Running doxygen in {repository_name}...")
    doxygen_result = subprocess.run(["doxygen"], cwd=repository_name)
    if doxygen_result.returncode != 0:
        raise RuntimeError(f"doxygen command failed")

    return repository_name

if __name__ == "__main__":
    print("=" * 72)
    print("Building unified FMSCoupler collection")
    print("=" * 72)
    
    clone_repository_and_run_doxygen("https://github.com/mlee03", "fmscoupler", "doc/all-round1")

    # Get documents from both sources without creating separate databases
    print("parse Gathering readmes...")
    readme_docs, readme_ids = document_readmes.build(docs_dir=DOCS_DIR, create_database=False)
    print(f"  -> {len(readme_docs)} documents from readmes")
    
    print("parse module documentation...")
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
   # print(f"\n[test] Testing unified collection...")
   # test_collection(COLLECTION_NAME, TESTS, LOG_FILE)
    
   # print(f"\nDone. Unified collection available:")
   # print(f"  {COLLECTION_NAME:<24}  log: {LOG_FILE}")

