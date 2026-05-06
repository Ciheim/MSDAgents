import ast
from pathlib import Path

class Document():

    def __init__(self, modulefilename):

        self.modulefilename = modulefilename
        self.modulename = self.parse_modulename()
        self.documents = {"functions": {}, "overview": None}
        self.rawcontent = None


    def read_file(self):
        with open(self.modulefilename, "r") as f:
            self.rawcontent = f.read()


    def parse_modulename(self):
        return Path(self.modulefilename).stem


    def parse_top_docstring(self) -> str | None:

        """
        Parses top-level documentation
        """

        tree = ast.parse(self.rawcontent)        
        docstring = ast.get_docstring(tree)
        
        return self._check(docstring)


    def parse_params(self, params_docstring: str) -> list[str]|str:
        """
        parses docstrings such as

        :param src_dir: is the absolute directory path to git clone the source code
        :type src_dir: str

        and saves the content as the two sentences below:

        src_dir is the absolute directory path to git clone th source code.
        src_dir is of type str
        """
        
        if not params_docstring: return ""
        
        params = []
        for param_and_type_string in params_docstring.split(":param"):
            paramstuff, typestring = param_and_type_string.split(":type", 1)
            paramname, paramstring = paramstuff.split(":", 1)
            params.append(f"""
                {paramname.strip()} {paramstring.strip()}.
                {paramname.strip()} is of type {typestring.strip()}.
            """)

        return params
    

    def parse_raises(self, raises_docstring) -> list[str]|str:
        """
        parses docstrings such as 

        :raises ValueError: Error if platform does not exist in platforms.yaml

        and saves the content as below:

        ValueError is raised if platform does not exist in platforms.yaml.
        """

        if not raises_docstring: return ""
        
        raises = []
        for a_docstring in raises_docstring.split(":raises"):
            raise_type, raise_condition = a_docstring.split(":", 1)
            raises.append(f"""
                {raise_type.strip()} is raised when {raise_condition.strip()}.
            """)

        return raises

    def parse_notes(self, notes_docstring) -> str:

        """
        parses docstrings such as 

        .. note:: This is some note
        
        and saves the content as below:

        This is some note.        
        """
        
        if not notes_docstring: return ""

        return notes_docstring.strip()


    def parse_docstring(self, docstring_in):
        
        docstring = self._check(docstring_in)
        overview = notes = raises = params = ""

        if docstring:
            if ".. note::" in docstring:
                docstring, notes = docstring.split(".. note::", 1)
            if ":raises" in docstring:
                docstring, raises = docstring.split(":raises", 1)
            if ":param" in docstring:
                docstring, params = docstring.split(":param", 1)

        return {
            "overview": self._check(docstring),
            "raises": self._check(raises),
            "notes": self._check(notes),
            "params": self._check(params)

        }
        

    def docstrings_to_sentences(self):

        """ 
        Parses the module file and saves docstrings into self.documents
        """

        self.documents["overview"] = self.parse_top_docstring()

        tree = ast.parse(self.rawcontent)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):

                docstring = ast.get_docstring(node)
                
                docstring_dict = self.parse_docstring(docstring)

                self.documents["functions"][node.name] = {
                    "overview": docstring_dict["overview"],
                    "params": self.parse_params(docstring_dict["params"]),
                    "raises": self.parse_raises(docstring_dict["raises"]),
                    "notes": self.parse_notes(docstring_dict["notes"])
                }


    def _check(self, docstring):
        """
        Utility method
        """
        if docstring is not None:
            docstring_stripped = docstring.strip()
            if docstring_stripped:
                return docstring_stripped        
        return ""

