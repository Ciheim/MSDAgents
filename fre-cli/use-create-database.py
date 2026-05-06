from create_database import ModuleDocument

makefiles = [
  "create_checkout_script.py",
  "create_compile_script.py",
  "create_docker_script.py",
  "create_makefile_script.py",
  "make_helpers.py",
  "run_fremake_script.py"
]

documents = {}
for myfile in makefiles:
    document = ModuleDocument(modulefilename=myfile, moduledir="files/")
    document.read_file()
    document.docstrings_to_sentences()
    print(document)
    documents[document.modulefilename] = document.documents    

