import importlib
import inspect
from pathlib import Path

import chromadb

import click

from fre import fre

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
]


for mod in modules:

    ids, documents, metadatas = [], [], []

    importedmod = importlib.import_module(mod)
    modfile = Path(inspect.getfile(importedmod)).name
    
    # module docstring
    
    mod_docstring = inspect.getdoc(importedmod)
    
    # each function is a document
    for functionname, function_obj in inspect.getmembers(importedmod):
        if inspect.isfunction(function_obj) and function_obj.__module__ == mod:

            print(mod, functionname)

            function_docstrings = function_obj.__doc__.split("\n")
            function_description = ""

            for iline in function_docstrings:
                iline_stripped = iline.strip()
                if iline_stripped:
                    if iline_stripped[0] == ":":
                        if ":param" in iline:
                            function_description += iline.replace(":", "")
                        elif "note::" in iline:
                            function_description += iline.replace("..", "").replace("::", ",")
                    else:
                        function_description += iline_stripped

            ids.append(functionname)
            documents.append(mod_docstring + function_description)
            metadatas.append({"mod": mod, "function": functionname, "source":  modfile})
            
mod = "fre.make.fremake"
functions = ["all", "checkout_script", "makefile", "compile_script", "dockerfile"]

importedmod = importlib.import_module(mod)
mod_docstring = importedmod.__doc__

for functionname, function_obj in inspect.getmembers(importedmod):
    if functionname in functions:
        description = f"Here is the information for fre make subcommand {functionname}:\n"
        with click.Context(function_obj) as ctx:
            description += ctx.get_help()
        ids.append(functionname)
        documents.append(description)
        metadatas.append({"mod": mod, "function": functionname, "source": "fremake.py"})
    
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
    for metadata in metadatas: print(metadata)
    for i_id in ids: print(i_id)

    
    
    


