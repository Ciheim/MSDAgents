import importlib
import inspect
import re
from types import SimpleNamespace

import click        

class ModuleDocument():

    def __init__(self, modulename: str, packagename: str = "fre"):
        """
        Class to convert docstrings in a module to a report format
        Example use case:
        self.packagename = "fre"
        self.modulename = "fre.make.fremake"
        self.documentation.overview = "This module contains functions to run the fremake script."
        self.documentation.functions = {
            "run_fremake_script": {
            "overview": "This function runs the fremake script."
            "params": [
                "src_dir is the absolute directory path to git clone the source code. src_dir is of type str."
            ],
            "raises": [
                "ValueError is raised when platform does not exist in platforms.yaml."
            ],
            "notes": "Note: This function is a wrapper around the fremake script."
            }
        }
        self.documentation.overview = "This module contains functions to run the fremake script."
        self.documentation.functions = {
            "run_fremake_script": "This function runs the fremake script.
             The function has the following parameters: src_dir is the absolute directory path to git clone the source code. 
             src_dir is of type str. 
             The function can raise the following exceptions: ValueError is raised when platform does not exist in platforms.yaml. 
             Note: This function is a wrapper around the fremake script."
        }
        self.metadata = {
            "run_fremake_script": {
                "name": "run_fremake_script",
                "module": "fre.make.fremake",
                "package": "fre"
            }
        }
        """
        self.packagename = packagename
        self.modulename = modulename
        self.metadata = {} 
        self.overview = None
        self.doc_dict = {} 
        self.documentation = SimpleNamespace(
            overview = None,
            functions = None
        )

        #sys.path.insert(0, str(Path(modulename).parent))
        self.mod = importlib.import_module(self.modulename)

    def docstrings_to_report(self):
        """
        Parses the module file and saves docstrings into this object.
        """

        self.documentation.overview = self._check(inspect.getdoc(self.mod))
        self.documentation.functions = {}

        for name, func in inspect.getmembers(self.mod, inspect.isfunction):
            if inspect.getmodule(func) is not self.mod:
                continue
                
            docstring = inspect.getdoc(func)
            docstring_dict = self.split_docstring(docstring)        

            doc_dict= {
                "overview": self._clean(docstring_dict["overview"]),
                "params": self.parse_params(docstring_dict["params"]),
                "raises": self.parse_raises(docstring_dict["raises"]),
                "notes": self.parse_notes(docstring_dict["notes"])
            }
            self.doc_dict[name] = doc_dict
            self.documentation.functions[name] = self.convert_to_reports(doc_dict)
            self.metadata[name] = {
                "name": name,
                "module": self.modulename,
                "package": self.packagename
            }

    def convert_to_reports(self, doc_dict):
        """
        Converts the sentences in self.functions into a report format.
        """
        
        report = doc_dict["overview"]
        if doc_dict["params"]:
            report += " The function has the following parameters: "
            for param in doc_dict["params"]:
                report += param
        if doc_dict["raises"]:
            report += " The function can raise the following exceptions: "
            for raise_ in doc_dict["raises"]:
                report += raise_
        if doc_dict["notes"]:
            report += " Note: " + doc_dict["notes"]
            
        return report

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
            type_ = typestring.split(":", 1)[1]
            
            paramname = self._check(paramname)
            params.append(
                f"{paramname} {self._clean(paramstring)}.  "
                f"{paramname} is of type {self._clean(type_)}.  "
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
        string_in = re.sub(r' {3,}', '  ', string_in)
        return string_in.strip()

    def __repr__(self):
        functions = []
        for name, report in self.documentation.functions.items():
            functions.append(f"* {name}:\n{report}\n")

        functions_text = "\n".join(functions)

        return (
            "__________________________________\n"
            f"MODULE:\n{self.modulename}\n\n"
            f"Overview:\n{self.documentation.overview}\n\n"
            f"Functions:\n{functions_text}"
            "__________________________________"
        )


class CommandDocument():

    def __init__(self, modulename, groupcommand: str):

        self.modulename = modulename
        self.groupcommand = groupcommand
        self.functions = {}
        self.overview = None

    def get_click_commands(self) -> list[str]:
        """
        Returns a list of Click command names registered on the Click group
        defined in this module.
        """
        mod = importlib.import_module(self.modulename)
        self.functions = {}
        
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, click.Command): #and obj.name == self.groupcommand:
                docstring = inspect.getdoc(obj)
                ctx = click.Context(obj)    
                help_output = obj.get_help(ctx)
                self.functions[obj.name] = {
                    "overview": self._clean(self._check(docstring)),
                    "help": self._check(help_output)
                }

        return list(self.functions.keys())
    
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
        string_in = re.sub(r' {3,}', '  ', string_in)
        return string_in.strip()
    
    def __repr__(self):
        commands_str = ""
        for name, content in self.functions.items():
            commands_str += f"  {name}:\n"
            commands_str += f"    Overview: {content['overview']}\n"
            commands_str += f"    Help: {content['help']}\n\n"
        
        return (
            f"COMMAND GROUP: {self.groupcommand}\n"
            f"Commands:\n{commands_str}"
        )