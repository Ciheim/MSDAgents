import copy

import fre
from documents import ModuleDocument, CommandDocument

class freDatabase:

    TOOLS_DICT = {
        "make": {
            "fre.make.create_checkout_script": ModuleDocument|None,
            "fre.make.create_compile_script": ModuleDocument|None,
            "fre.make.create_docker_script": ModuleDocument|None,
            "fre.make.create_makefile_script": ModuleDocument|None,
            "fre.make.make_helpers": ModuleDocument|None,
            "fre.make.run_fremake_script": ModuleDocument|None
        },
        "yaml": {
            "fre.yamltools.abstract_classes": ModuleDocument|None,
            "fre.yamltools.combine_yamls_script": ModuleDocument|None,
            "fre.yamltools.constructors": ModuleDocument|None,
            "fre.yamltools.helpers": ModuleDocument|None,
        },
        "app": {
            "fre.app.generate_time_averages.cdoTimeAverager": ModuleDocument|None,
            "fre.app.generate_time_averages.combine": ModuleDocument|None,
            "fre.app.generate_time_averages.frenctoolsTimeAverager": ModuleDocument|None,
            "fre.app.generate_time_averages.frepytoolsTimeAverager": ModuleDocument|None,
            "fre.app.generate_time_averages.generate_time_averages": ModuleDocument|None,
            "fre.app.generate_time_averages.timeAverager": ModuleDocument|None,
            "fre.app.generate_time_averages.wrapper": ModuleDocument|None,
            "fre.app.mask_atmos_plevel.mask_atmos_plevel": ModuleDocument|None,
            "fre.app.regrid_xy.regrid_xy": ModuleDocument|None,
            "fre.app.remap_pp_components.remap_pp_components": ModuleDocument|None,
        },
        "list": {
            "fre.list_.list_experiments_script": ModuleDocument|None,
            "fre.list_.list_platforms_script": ModuleDocument|None,
            "fre.list_.list_pp_components_script": ModuleDocument|None,
        },
        "pp": {
            "fre.pp.checkout_script": ModuleDocument|None,
            "fre.pp.configure_script_yaml": ModuleDocument|None,
            "fre.pp.histval_script": ModuleDocument|None,
            "fre.pp.install_script": ModuleDocument|None,
            "fre.pp.nccheck_script": ModuleDocument|None,
            "fre.pp.ppval_script": ModuleDocument|None,
            "fre.pp.rename_split_script": ModuleDocument|None,
            "fre.pp.run_script": ModuleDocument|None,
            "fre.pp.split_netcdf_script": ModuleDocument|None,
            "fre.pp.status_script": ModuleDocument|None,
            "fre.pp.trigger_script": ModuleDocument|None,
            "fre.pp.validate_script": ModuleDocument|None,
            "fre.pp.wrapper_script": ModuleDocument|None
        },
        "run": {
            "fre.run.frerunexample": ModuleDocument|None
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
        self.tools_dict = copy.deepcopy(self.TOOLS_DICT)
        self.commands_dict = copy.deepcopy(self.COMMANDS_DICT)

    def summarize(self):
        for tool, tool_dict in self.tools_dict.items():
            for modulename in tool_dict.keys():
                try:
                    moduledocument = ModuleDocument(modulename)
                    moduledocument.summarize()
                    self.tools_dict[tool][modulename] = moduledocument
                except Exception as exc:
                    print(f"Skipping module {modulename}: {exc}")
        
        for command, command_dict in self.commands_dict.items():
            try:
                commanddocument = CommandDocument(command)
                commanddocument.summarize()
                self.commands_dict[command] = commanddocument
            except Exception as exc:
                print(f"Skipping command group {command}: {exc}")
        
    def to_chromadb(self):
        # Convert the summarized data into a format suitable for ChromaDB
        document_list = []
        metadata_list = []
        id_list = []

        for tool, tool_dict in self.tools_dict.items():
            for modulename, moduledocument in tool_dict.items():
                if moduledocument is not None:
                    for function in moduledocument.functions:
                        document_list.append(moduledocument.overview + moduledocument.functions[function])
                        metadata_list.append(moduledocument.metadata[function])
                        id_list.append(f"{modulename}.{function}")
        
            for command, commanddocument in self.commands_dict.items():
                if commanddocument is not None:
                    for subcommand in commanddocument.commands:
                        document_list.append(commanddocument.overview + commanddocument.commands[subcommand])
                        metadata_list.append(commanddocument.metadata)
                        id_list.append(f"{command}.{subcommand}")         
        
        return document_list, metadata_list, id_list

fre = freDatabase()
fre.summarize()
document_list, metadata_list, id_list = fre.to_chromadb()
