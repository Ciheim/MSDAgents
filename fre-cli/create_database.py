import importlib.util
import inspect
import click
from pathlib import Path

class ModuleDocument():

    def __init__(self, modulefilename: str|Path, moduledir: str|Path = "./"):

        self.moduledir = Path(moduledir)
        self.modulefilename = Path(modulefilename)
        self.modulename = self.modulefilename.stem
        self.documents = {"functions": {}, "overview": None}

        if not (self.moduledir/self.modulefilename).exists():
            raise RuntimeError(f"Cannot find {modulefilename} in {moduledir}")

        spec = importlib.util.spec_from_file_location(
            self.modulename, self.moduledir / self.modulefilename
        )
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def parse_params(self, params_docstring: str) -> list[str]|str:
        """
        parses docstrings such as
        ```
        :param src_dir: is the absolute directory path to git clone the source code
        :type src_dir: str
        ```
        and saves the content as the two sentences below:
        ```
        src_dir is the absolute directory path to git clone th source code.
        src_dir is of type str
        ```
        """
        
        if not params_docstring: return ""
        
        params = []
        for param_and_type_string in params_docstring.split(":param"):
            paramstuff, typestring = param_and_type_string.split(":type", 1)
            paramname, paramstring = paramstuff.split(":", 1)
            
            paramname = self._clean(paramname)     
            params.append(
                f"{paramname} {self._clean(paramstring)}. "
                f"{paramname} is of type {self._clean(typestring)}."
            )

        return params
    
    def parse_raises(self, raises_docstring) -> list[str]|str:
        """
        parses docstrings such as 
        ```
        :raises ValueError: Error if platform does not exist in platforms.yaml
        ```
        and saves the content as below:
        ```
        ValueError is raised if platform does not exist in platforms.yaml.
        ```
        """

        if not raises_docstring: return ""
        
        raises = []
        for a_docstring in raises_docstring.split(":raises"):
            raise_type, raise_condition = a_docstring.split(":", 1)
            raises.append(
                f"{self._clean(raise_type)} is raised when {self._clean(raise_condition)}."
            )

        return raises

    def parse_notes(self, notes_docstring) -> str:
        """
        parses docstrings such as 
        ```
        .. note:: This is some note
        ```
        and saves the content as below:
        ````
        This is some note.     
        ```   
        """
        
        if not notes_docstring: return ""

        return self._clean(notes_docstring)


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

        self.documents["overview"] = self._check(inspect.getdoc(self.mod))

        for name, func in inspect.getmembers(self.mod, inspect.isfunction):
            if inspect.getmodule(func) is not self.mod:
                continue
            docstring_dict = self.parse_docstring(inspect.getdoc(func))
            self.documents["functions"][name] = {
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
    
    def _clean(self, string_in):
        for escape_char in ("\n", "\t", "\r", "\b", "\f", "\v", "\0"):
            string_in = string_in.replace(escape_char, "")
        return string_in.strip()

    def __repr__(self):
        functions = "\n".join(
            f"  {name}:\n"
            f"    Overview: {content['overview']}\n"
            f"    Params: {content['params']}\n"
            f"    Raises: {content['raises']}\n"
            f"    Notes: {content['notes']}\n\n"
            for name, content in self.documents["functions"].items()
        )
        return (
            f"Overview:\n{self.documents['overview']}\n"
            f"Functions:\n{functions}"
        )


class CommandDocument():

    def __init__(self, filename, filedir: str|Path = Path("./")):

        self.filename = Path(filename)
        self.filedir = Path(filedir)
        self.modulename = self.filename.stem
        self.rawcontent = None

    def read_file(self):
        with open(Path(self.filedir)/self.filename, "r") as f:
            self.rawcontent = f.read()


    def get_click_commands(self) -> list[str]:
        """
        Returns a list of Click command names found in self.filename.
        """
        spec = importlib.util.spec_from_file_location(
            self.filename.stem, Path(self.filedir) / self.filename
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        return [
            name for name in dir(mod)
            if isinstance(getattr(mod, name), click.BaseCommand)
        ]
    

