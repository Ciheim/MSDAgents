from create_database import ModuleDocument, CommandDocument
#import ChromaDB

makemodules = [
  "fre.make.create_checkout_script",
  "fre.make.create_compile_script",
  "fre.make.create_docker_script",
  "fre.make.create_makefile_script",
  "fre.make.make_helpers",
  "fre.make.run_fremake_script"
]
yamlmodules = [
    "fre.yamltools.abstract_classes",
    "fre.yamltools.combine_yamls_script",
    "fre.yamltools.constructors",
    "fre.yamltools.helpers",

]
appmodules = [
    "fre.app.generate_time_averages.cdoTimeAverager",
    "fre.app.generate_time_averages.combine",
    "fre.app.generate_time_averages.frenctoolsTimeAverager",
    "fre.app.generate_time_averages.frepytoolsTimeAverager",
    "fre.app.generate_time_averages.generate_time_averages",
    "fre.app.generate_time_averages.timeAverager",
    "fre.app.generate_time_averages.wrapper",
    "fre.app.mask_atmos_plevel.mask_atmos_plevel",
    "fre.app.regrid_xy.regrid_xy",
    "fre.app.remap_pp_components.remap_pp_components",
]
listmodules = [
    "fre.list_.list_experiments_script",
    "fre.list_.list_platforms_script",
    "fre.list_.list_pp_components_script",    
]
ppmodules = [
    "fre.pp.checkout_script",
    "fre.pp.configure_script_yaml",
    "fre.pp.histval_script",
    "fre.pp.install_script",
    "fre.pp.nccheck_script",
    "fre.pp.ppval_script",
    "fre.pp.rename_split_script",
    "fre.pp.run_script",
    "fre.pp.split_netcdf_script",
    "fre.pp.status_script",
    "fre.pp.trigger_script",
    "fre.pp.validate_script",
    "fre.pp.wrapper_script"
]
runmodules = [
    "fre.run.frerunexample"
]

makecommands = [{"modulename": "fre.make.fremake", "groupcommand": "fre make"}]
yamlcommands = [{"modulename": "fre.yamltools.freyamltools", "groupcommand": "fre yamltools"}]
appcommands = [{"modulename": "fre.app.freapp", "groupcommand": "fre app"}]
catalogcommands = [{"modulename": "fre.catalog.frecatalog", "groupcommand": "fre catalog"}]
listcommands = [{"modulename": "fre.list_.frelist", "groupcommand": "fre list"}]
ppcommands = [{"modulename": "fre.pp.frepp", "groupcommand": "fre pp"}]
runcommands = [{"modulename": "fre.run.frerun", "groupcommand": "fre run"}]

#schemas
#mkmf

allmodules = [
    makemodules,
    yamlmodules,
    appmodules,
    listmodules,
    ppmodules,
    runmodules
]
allcommands =[
    makecommands,
    yamlcommands,
    appcommands,
    catalogcommands,
    listcommands,
    ppcommands,
    runcommands
]

documents = {}
for module_group in allmodules:
    for amodule in module_group:
        try:
            document = ModuleDocument(amodule)
            document.summarize()
            documents[amodule] = document
        except Exception as exc:
            print(f"Skipping module {amodule}: {exc}")

clickdocuments = {}
for command_group in allcommands:
    for command in command_group:
        try:
            document = CommandDocument(**command)
            document.summarize()
            clickdocuments[command["modulename"]] = document
        except Exception as exc:
            print(f"Skipping command group {command['modulename']}: {exc}")
