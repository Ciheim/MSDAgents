import importlib
import inspect
import re
from typing import TypedDict

import click


class MetadataEntry(TypedDict):
    name: str
    module: str
    package: str


class Document():

    def __init__(self, modulename: str, packagename: str = "fre"):
        self.packagename = packagename
        self.modulename = modulename
        self.metadata: dict[str, MetadataEntry] = {}
        self.docstring_dict = {}

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


class ModuleDocument(Document):

    """
    Class to convert docstrings in a module to a report format
    Example use case:
    self.packagename = "fre"
    self.modulename = "fre.make.create_checkout_script"
    self.documentation.overview = "top level docstring for the module"
    self.dict_ = {
        "baremetal_checkout_write": {
        "overview": "baremetal_checkout_write is called by checkout_create..."
        "params": [
            "model_yaml is a 'freyaml' class object containing.. .model_yaml is of type yamlfre.freyaml",
            "src_dir is the absolute directory path to git clone the source code. src_dir is of type str."
        ],
        "raises": [
            "ValueError is raised when platform does not exist in platforms.yaml."
        ],
        "notes": "This is the docstring starting with .. note::"
        }
    }
    self.summary.overview = "top level docstring for the module"
    self.summary.functions = {"baremetal_checkout_write": 
        "baremetal_checkout_write is called by checkout_create...
            The function has the following parameters: model_yaml is a 'freyaml' class object containing.. .
            model_yaml is of type yamlfre.freyaml. 
            src_dir is the absolute directory path to git clone the source code. src_dir is of type str.
            The following errors can be raised: ValueError is raised when platform does not exist in platforms.yaml. 
            Note: This is the docstring starting with .. note::"
    }
    self.metadata = {
        "baremetal_checkout_write": {
            "name": "baremetal_checkout_write",
            "module": "fre.make.create_checkout_script",
            "package": "fre"
        }
    }
    """

    def __init__(self, modulename: str, packagename: str = "fre"):
        """Constructor"""
        super().__init__(modulename, packagename)
        self.overview = None
        self.functions = {}

        #sys.path.insert(0, str(Path(modulename).parent))
        self.mod = importlib.import_module(self.modulename)

    def summarize(self):
        """
        Parses the module file and saves docstrings into this object.
        """

        #set module overview
        self.overview = self._check(inspect.getdoc(self.mod))
        
        #initialize dictionary to hold function documentation
        self.functions = {}

        #for each function in module
        for name, func in inspect.getmembers(self.mod, inspect.isfunction):
            if inspect.getmodule(func) is not self.mod:
                continue

            #get function docstring 
            docstring = inspect.getdoc(func)
            
            #split docstring into overview, params, raises, and notes
            parsed_docstring = self.split_docstring(docstring)

            #save
            docstring_dict= {
                "overview": self._clean(parsed_docstring["overview"]),
                "params": self.parse_params(parsed_docstring["params"]),
                "raises": self.parse_raises(parsed_docstring["raises"]),
                "notes": self.parse_notes(parsed_docstring["notes"])
            }
            
            #store
            self.docstring_dict[name] = docstring_dict
            
            #convert doc_dict to paragraphsn
            self.functions[name] = self.summarize_docstring_dict(docstring_dict)
            
            #save metadata 
            self.metadata[name] = {
                "name": name,
                "module": self.modulename,
                "package": self.packagename
            }

    def summarize_docstring_dict(self, docstring_dict):
        """
        Converts the sentences in self.functions into a report format.
        """
        
        summary = docstring_dict["overview"]
        if docstring_dict["params"]:
            summary += " The function has the following parameters: "
            for param in docstring_dict["params"]:
                summary += param
        if docstring_dict["raises"]:
            summary += " The function can raise the following exceptions: "
            for raise_ in docstring_dict["raises"]:
                summary += raise_
        if docstring_dict["notes"]:
            summary += " Note: " + docstring_dict["notes"]
            
        return summary

    def split_docstring(self, docstring_in):
        """
        Splits a function docstring into overview, params, raises, and notes
        """
        
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

    def __repr__(self):
        functions = []
        for name, report in self.functions.items():
            functions.append(f"* {name}:\n{report}\n")

        functions_text = "\n".join(functions)

        return (
            "__________________________________\n"
            f"MODULE:\n{self.modulename}\n\n"
            f"Overview:\n{self.overview}\n\n"
            f"Functions:\n{functions_text}"
            "__________________________________"
        )


class CommandDocument(Document):    

    """
    Class to convert docstrings in a module to a report format for Click commands.
    Example use case:
    self.packagename = "fre"
    self.modulename = "fre.make.fremake"
    self.groupcommand = "fre make"
    self.docstring_dict = {
        "all": {
            "overview": "all is the main command that calls all other commands in the group.",
            "help": "output from fre make all --help"
        }
    }
    self.summary.overview = "top level docstring for the module"
    self.summary.commands = {"all":
        "all is the main command that calls all other commands in the group.
            The help information for this command is as follows: ..."
    }
    self.metadata = {
        "all": {
            "name": "all",
            "module": "fre.make.fremake",
            "package": "fre"
        }
    }
    """

    def __init__(self, modulename: str, groupcommand: str = None, packagename: str = "fre"):
        """Constructor"""
        super().__init__(modulename, packagename)
        self.groupcommand = groupcommand
        self.overview = None
        self.commands = {}

    def summarize(self):
        """
        Parses the module file and saves Click command docstrings into this object.
        """
        mod = importlib.import_module(self.modulename)

        #set module overview
        self.overview = self._check(inspect.getdoc(mod))

        #initialize dictionary to hold command documentation
        self.commands = {}

        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, click.Command): #and obj.name == self.groupcommand:
                docstring = inspect.getdoc(obj)
                ctx = click.Context(obj)
                help_output = obj.get_help(ctx)

                #save
                command_dict = {
                    "overview": self._clean(self._check(docstring)),
                    "help": self._check(help_output)
                }

                #store
                self.docstring_dict[name] = command_dict

                #convert command_dict to paragraph
                self.commands[name] = self.summarize_docstring_dict(command_dict)

                #save metadata
                self.metadata[name] = {
                    "name": name,
                    "module": self.modulename,
                    "package": self.packagename
                }

    def summarize_docstring_dict(self, command_dict):
        """
        Converts the sentences in self.commands into a report format.
        """
        report = (
            f"{command_dict['overview']} "
            f"The help information for this command is as follows: {command_dict['help']}"
        )
        return report

    def __repr__(self):
        commands = []
        for name, report in self.commands.items():
            commands.append(f"* {name}:\n{report}\n")

        commands_text = "\n".join(commands)

        return (
            "__________________________________\n"
            f"MODULE:\n{self.modulename}\n\n"
            f"Overview:\n{self.overview}\n\n"
            f"Commands:\n{commands_text}"
            "__________________________________"
        )