import importlib
import inspect
from pathlib import Path

from bs4 import BeautifulSoup
import chromadb

import fre
import fre.make

#create database collection
db_path = "./fremake_database"

#use default sentence-transformer embedding function 
client = chromadb.PersistentClient(path=db_path)
collection_name = "fremake"
collection_metadata = {"version": "1.1.1", "module": "make"}
collection = client.get_or_create_collection(name = collection_name, metadata = collection_metadata)

#read data
modules =  [
    "fre.make.create_checkout_script",
    "fre.make.create_compile_script",
    "fre.make.create_docker_script",
    "fre.make.create_makefile_script",
    "fre.make.make_helpers",
    "fre.make.run_fremake_script"
    #fre.make.fremake needs special treatment!
    #"fre.make.fremake",
]


for mod in modules:

    ids, documents, metadatas = [], [], []

    importedmod = importlib.import_module(mod)
    modfile = Path(inspect.getfile(importedmod)).name
    
    # module docstring
    moddoc = importedmod.__doc__ 
    
    # each function is a document
    for functionname, function_obj in inspect.getmembers(importedmod):
        if inspect.isfunction(function_obj) and function_obj.__module__ == mod:
            print(function_obj.__module__, functionname)
            ids.append(functionname)
            documents.append(moddoc + function_obj.__doc__)
            metadatas.append({"mod": mod, "function": functionname, "source":  modfile})
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            

## testing
testme = False
if testme:
    collections = client.list_collections()
    #collections = [client.get_collection(name=collection) for collection.name in collection_list]

    collection0 = collections[0]
    getresults = collection0.get()
    ids, documents, metadatas = getresults["ids"], getresults["documents"], getresults["metadatas"]

    for document in documents: print(document)
#    for i_id in ids: print(i_id)
#    for metadata in metadatas: print(metadata)
    
    
    


