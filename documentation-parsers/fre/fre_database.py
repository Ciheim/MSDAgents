import copy

import fre
from documents import ModuleDocument, CommandDocument

class FreDatabase:

    TOOLS_DICT: dict[str, dict[str, ModuleDocument | None]] = {
        "make": {
            "fre.make.create_checkout_script": None,
            "fre.make.create_compile_script": None,
            "fre.make.create_docker_script": None,
            "fre.make.create_makefile_script": None,
            "fre.make.make_helpers": None,
            "fre.make.run_fremake_script": None
        },
        "yaml": {
            "fre.yamltools.abstract_classes": None,
            "fre.yamltools.combine_yamls_script": None,
            "fre.yamltools.constructors": None,
            "fre.yamltools.helpers": None,
        },
        "app": {
            "fre.app.generate_time_averages.cdoTimeAverager": None,
            "fre.app.generate_time_averages.combine": None,
            "fre.app.generate_time_averages.frenctoolsTimeAverager": None,
            "fre.app.generate_time_averages.frepytoolsTimeAverager": None,
            "fre.app.generate_time_averages.generate_time_averages": None,
            "fre.app.generate_time_averages.timeAverager": None,
            "fre.app.generate_time_averages.wrapper": None,
            "fre.app.mask_atmos_plevel.mask_atmos_plevel": None,
            "fre.app.regrid_xy.regrid_xy": None,
            "fre.app.remap_pp_components.remap_pp_components": None,
        },
        "list": {
            "fre.list_.list_experiments_script": None,
            "fre.list_.list_platforms_script": None,
            "fre.list_.list_pp_components_script": None,
        },
        "pp": {
            "fre.pp.checkout_script": None,
            "fre.pp.configure_script_yaml": None,
            "fre.pp.histval_script": None,
            "fre.pp.install_script": None,
            "fre.pp.nccheck_script": None,
            "fre.pp.ppval_script": None,
            "fre.pp.rename_split_script": None,
            "fre.pp.run_script": None,
            "fre.pp.split_netcdf_script": None,
            "fre.pp.status_script": None,
            "fre.pp.trigger_script": None,
            "fre.pp.validate_script": None,
            "fre.pp.wrapper_script": None
        },
        "run": {
            "fre.run.frerunexample": None
        },
    }

    COMMANDS_DICT = {
        "fre.make.fremake": CommandDocument|None,
        "fre.yamltools.freyamltools": CommandDocument|None,
        "fre.app.freapp": CommandDocument|None,
        "fre.catalog.frecatalog": CommandDocument|None,
        "fre.list_.frelist": CommandDocument|None,
        "fre.pp.frepp": CommandDocument|None,
        "fre.run.frerun": CommandDocument|None
    }

    #commandcontent = {
    #    "make": [{"modulename": "fre.make.fremake", "groupcommand": "fre make"}],
    #    "yaml": [{"modulename": "fre.yamltools.freyamltools", "groupcommand": "fre yamltools"}],
    #    "app": [{"modulename": "fre.app.freapp", "groupcommand": "fre app"}],
    #    "catalog": [{"modulename": "fre.catalog.frecatalog", "groupcommand": "fre catalog"}],
    #    "list": [{"modulename": "fre.list_.frelist", "groupcommand": "fre list"}],
    #    "pp": [{"modulename": "fre.pp.frepp", "groupcommand": "fre pp"}],
    #    "run": [{"modulename": "fre.run.frerun", "groupcommand": "fre run"}],
    #}

    def __init__(self):
        """Constructor"""
        self.tools_dict = copy.deepcopy(self.TOOLS_DICT)
        self.commands_dict = copy.deepcopy(self.COMMANDS_DICT)

    def summarize(self):
        """Summarizes the docstrings in the modules and commands"""
        for tool, tool_dict in self.tools_dict.items():
            for modulename in tool_dict.keys():
                try:
                    moduledocument = ModuleDocument(modulename)
                    moduledocument.summarize()
                    self.tools_dict[tool][modulename] = moduledocument
                except Exception as exc:
                    raise RuntimeError(f"Skipping module {modulename}") from exc
        
        for command, command_dict in self.commands_dict.items():
            try:
                commanddocument = CommandDocument(command)
                commanddocument.summarize()
                self.commands_dict[command] = commanddocument
            except Exception as exc:
                raise RuntimeError(f"Skipping command group {command}") from exc
        
    def to_chromadb(self):
        """Converts the summarized data into document_list, metadata_list, and id_list for ChromaDB"""
        document_list = []
        metadata_list = []
        id_list = []

        for tool, tool_dict in self.tools_dict.items():
            for modulename, moduledocument in tool_dict.items():
                for function in moduledocument.functions:
                    document_list.append(moduledocument.module_overview + moduledocument.functions[function])
                    metadata_list.append(moduledocument.metadata[function])
                    id_list.append(f"{modulename}.{function}")
        
            for command, commanddocument in self.commands_dict.items():
                for subcommand in commanddocument.subcommands:
                    document_list.append(commanddocument.command_overview + commanddocument.subcommands[subcommand])
                    metadata_list.append(commanddocument.metadata)
                    id_list.append(f"{command}.{subcommand}")         
        
        return document_list, metadata_list, id_list

fre_db = FreDatabase()
fre_db.summarize()
document_list, metadata_list, id_list = fre_db.to_chromadb()
