import ast 

class MethodDocument():

    def __init__(self):
        self.modulename = None
        self.params = {}
        self.note = {}
        self.raises = {}

class Document():

    def __init__(self, modulefilename, modulename):

        self.modulefilename = modulefilename
        self.modulename = modulename
        self.documents = {"function": {}, "overview": None}
        self.rawcontent = None

    def read_file(self):
        with open(self.modulefilename, "r") as f:
            self.rawcontent = f.read()

    def get_top_docstring(self) -> str | None:
        tree = ast.parse(self.rawcontent)
        self.documents["overview"] = ast.get_docstring(tree)

    def parse_params(self, params_docstring: str) -> list[str]:
        params = []
        
        for astring in params_docstring.split(":param"):
            paramstuff, typestring = astring.split(":type", 1)
            paramname, paramstring = paramstuff.split(":param", 1)[1].split(":")
            params.append(f"""
                {paramname.strip()} {paramstring.strip()}.
                {paramname.strip()} is of type {tyepstring.strip()}.
            """)

        return params
    
    def parse_raises(self, raises_docstring) -> list[str]:
        raises = []

        for astring in raises_docstring(":raises"):
            raise_type, raise_condition = astring.split(":")
            raises.append("""
                {raise_type.strip()} is raised when {raise_condition.strip().}
            """)

    def parse_notes(self, notes_docstring) -> str:
        
        return notes_docstring.split(".. note::")[1].strip()


    def parse_docstring(self, docstring_in):
        
        docstring = docstring_in

        notes, raises, params = "", "", ""

        if "..note::" in docstring:
            docstring, notes = docstring.split(".. notes::", 1)
        if ":raises" in docstring:
            docstring, raises = docstring.split(":raises", 1)
        if ":param" in docstring:
            docstring, params = docstring, split("param", 1)

        return {"overview": docstring, "params": params, "notes": notes}


    def get_param_docstrings(self):
        tree = ast.parse(self.rawcontent)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functionname, functiondict = node.name, {}
                docstring = ast.get_docstring(node)

                docstring_dict = self.parse_docstring(docstring)
                self.documents["functions"][node.name] = 
                    {"overview": docstring_dict["overview"]
                     "params: self.parse_params(docstring_dict["params"])
                    "notes": self.parse_params(docstring_dict["notes"]]
                    }