import importlib
import importlib.util
import inspect
import sys
import click
from pathlib import Path

class ModuleDocument():

    def __init__(self, modulename: str, packagename: str = "fre"):

        self.packagename = packagename
        self.modulename = modulename
        self.documents = {"functions": {}, "overview": None}

        #sys.path.insert(0, str(Path(modulename).parent))
        self.mod = importlib.import_module(self.modulename)

    def docstrings_to_sentences(self):
        """
        Parses the module file and saves docstrings into self.documents
        """

        self.documents["overview"] = self._check(inspect.getdoc(self.mod))

        for name, func in inspect.getmembers(self.mod, inspect.isfunction):
            if inspect.getmodule(func) is not self.mod:
                continue

            docstring = inspect.getdoc(func)
            docstring_dict = self.split_docstring(docstring)        

            self.documents["functions"][name] = {
                "overview": self._clean(docstring_dict["overview"]),
                "params": self.parse_params(docstring_dict["params"]),
                "raises": self.parse_raises(docstring_dict["raises"]),
                "notes": self.parse_notes(docstring_dict["notes"])
            }

    def split_docstring(self, docstring_in):
        
        docstring = self._check(docstring_in)
        notes = raises = params = ""

        if docstring:
            if ".. note::" in docstring:
                docstring, notes = docstring.split(".. note::", 1)
            if ":raises" in docstring:
                docstring, raises = docstring.split(":raises", 1)
            if ":param" in docstring:
                docstring, params = docstring.split(":param", 1)

        return {
            "overview": docstring,
            "raises": raises,
            "notes": notes,
            "params": params
        }

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
        
        if not self._check(params_docstring): return ""
        
        params = []
        for param_and_type_string in params_docstring.split(":param"):
            paramstuff, typestring = param_and_type_string.split(":type", 1)
            paramname, paramstring = paramstuff.split(":", 1)
            
            paramname = self._check(paramname)
            params.append(
                f"{paramname} {self._clean(paramstring)}. "
                f"{paramname} is of type {self._clean(typestring)}."
            )

        return params
    
    def parse_raises(self, raises_docstring) -> list[str]|str:
        """
        parses docstrings such as ':raises ValueError: Error if platform does not exist in platforms.yaml'
        and saves the content as 'ValueError is raised if platform does not exist in platforms.yaml.'
        """

        if not self._check(raises_docstring): return ""
        
        raises = []
        for a_docstring in raises_docstring.split(":raises"):
            raise_type, raise_condition = a_docstring.split(":", 1)
            raises.append(
                f"{self._clean(raise_type)} is raised when {self._clean(raise_condition)}."
            )

        return raises

    def parse_notes(self, notes_docstring) -> str:
        """
        parses docstrings such as '.. note:: This is some note'        
        and saves the content as 'This is some note.'
        """    
        if not self._check(notes_docstring): return ""
        return self._clean(notes_docstring)

        
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
        """
        Utility method
        """
        for escape_char in ("\n", "\t", "\r", "\b", "\f", "\v", "\0"):
            string_in = string_in.replace(escape_char, "")
        return string_in.strip()

    def __repr__(self):
        functions = []
        for name, content in self.documents["functions"].items():
            params_str = "\n      ".join(content['params']) if content['params'] else ""
            raises_str = "\n      ".join(content['raises']) if content['raises'] else ""
            
            func_str = f"  {name}:\n"
            func_str += f"    Overview: {content['overview']}\n"
            func_str += f"    Params:\n      {params_str}\n" if params_str else "    Params:\n"
            func_str += f"    Raises:\n      {raises_str}\n" if raises_str else "    Raises:\n"
            func_str += f"    Notes: {content['notes']}\n\n"
            functions.append(func_str)
        
        return (
            f"MODULE: {self.modulename}\n"
            f"Overview:\n{self.documents['overview']}\n"
            f"Functions:\n{''.join(functions)}"
        )


class CommandDocument():

    def __init__(self, filename, filedir: str|Path = Path("./")):

        self.filename = Path(filename)
        self.filedir = Path(filedir)
        self.modulename = self.filename.stem

    def get_click_commands(self) -> list[str]:
        """
        Returns a list of Click command names found in self.filename.
        """
        mod = importlib.import_module(self.filename.stem)

        commands = []

        def collect_commands(command: click.Command, parent_path: str = "") -> None:
            command_path = f"{parent_path} {command.name}".strip()
            commands.append(command_path)

            if isinstance(command, click.Group):
                for subcommand_name in command.list_commands(click.Context(command)):
                    subcommand = command.get_command(click.Context(command), subcommand_name)
                    if subcommand is not None:
                        collect_commands(subcommand, command_path)

        for name in dir(mod):
            command = getattr(mod, name)
            if isinstance(command, click.Command):
                collect_commands(command)

        return commands

    



