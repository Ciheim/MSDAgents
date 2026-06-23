from pathlib import Path
from typing import Literal

from bs4 import BeautifulSoup

from langchain_core.documents import Document


class XMLsoup():

    def __init__(self, xmldir: str|Path = "./", xmlfile: str|Path = None):

        self.xmldir = xmldir
        self.xmlfile = xmlfile
        self.soup = self.get_xmlsoup()
        self.toplevel_name = self.get_name(toplevel=True)
        
    def get_xmlsoup(self):

        if self.xmlfile is None:
            raise IOError("xmlfile not specified")

        xmlfile = Path(self.xmldir)/Path(self.xmlfile)

        if xmlfile.exists():
            with open(xmlfile, "r") as openedfile:
                return BeautifulSoup(openedfile, "lxml-xml")
        else:
            raise FileNotFoundError(f"xml file '{self.xmlfile}' in directory '{self.xmldir}' does not exist")        

        
    def get_name(self, soup = None, toplevel = False):

        """
        Retrieves the name tag.
        The module name is a special case: <compoundname>atm_land_ice_flux_exchange.F90</compoundname>
        """
        
        if soup is None:
            soup = self.soup

        tag = "name"
        if toplevel:
            tag = "compoundname"

        name = self.get_tag_to_string(tag, soup)

        if name is not None:
            namstr = name.strip()
            if namstr:
                return namstr

        raise RuntimeError(f"cannot find tag '{tag}' to set name in {self.soup}.  ")
    
                
    
    def get_tag_to_string(self, tag, soup = None):

        """
        Returns the text of the tag
        For example, returns the text in
        <parblock> text </parblock>
        """

        if soup is None:
            soup = self.soup

        tagobj = soup.find(tag)                
        
        if tagobj is not None:
            tagstr = tagobj.text.strip()
            if tagstr:
                return tagstr
            
        return ""


    def get_parameters_description(self, soup = None, subroutine_name = None):

        """
        Returns the parameters for a subroutine as tables by parsing
        <parameterlist>
            <parameteritem>
                <parameternamelist>
                    <parametername direction="in">time</parametername>
                </parameternamelist>
                <parameterdescription>
                    <para>is the current model time</para>
                </parameterdescription>
            </parameteritem>
        </parameterlist>
        into 
        | Name | Type | Subroutine | Definition |
        |------|------|------------|------------|
        | time | intent(in) | subroutine_name | is the current model time |        
        """

        if soup is None:
            soup = self.soup

        parameter_item_objs = soup.find_all("parameteritem")

        if not parameter_item_objs:
            return None

        table = "| Name | Type | Subroutine | Definition |\n|------|------|------------|------------|\n"
        for parameter_item_obj in parameter_item_objs:
            parameter_namelist_obj = parameter_item_obj.parameternamelist
            if parameter_namelist_obj is None:
                raise RuntimeError(f"cannot find parameter namelist in {parameter_item_obj}")

            inout = parameter_namelist_obj.get("direction", "inout")
            parameter_name = parameter_namelist_obj.text.split()[0].strip()
            parameter_description = self.get_tag_to_string("parameterdescription", parameter_item_obj)

            table += f"| {parameter_name} | intent({inout}) | {subroutine_name or ''} | {parameter_description} |\n"

        return table

    def get_inbodydescription(self, soup = None):
        
        """
        Parses
        <inbodydescription>
        <para><parblock><para>INITIALIZE MODULE-LEVEL VARIABLES. </para></parblock></para>
        <para><parblock><para>GET FILE UNIT FOR STDOUT AND STDLOG FOR INTERNAL LOGGING PURPOSES </para></parblock></para>
        </inbodydescription>
        into
        Step 1: INITIALIZE MODULE-LEVEL VARIABLES.
        Step 2: GET FILE UNIT FOR STDOUT AND STDLOG FOR INTERNAL LOGGING PURPOSES
        """

        if soup is None:
            soup = self.soup

        inbodydescription_obj = soup.find("inbodydescription")

        if inbodydescription_obj is None:
            return ""

        steps = ""
        if inbodydescription_obj.text.strip():
            for istep, step in enumerate(inbodydescription_obj.find_all("parblock")):
                if step.text.strip():
                    steps += f"Step {istep+1}: {step.text.strip()}\n"

        return steps


class ModuleTopLevelDocument(XMLsoup):
    """Parse module-level documentation from Doxygen XML files.
    
    Extracts overview information from Fortran module documentation,
    combining brief descriptions and detailed descriptions from parblocks.
    """

    def __init__(self,
                 xmldir: str|Path = "./docs/xml",
                 xmlfile: str|Path = None):

        """
        Parses top-level documentation in files like atm__land__ice_flux__exchange__f90.xml
        to append to documentation from namespaceatm__land_ice_flux_exchange__f90.xml.
        Overview is expected to for example be:
        !! @brief Module atm_land_ice_flux_exchange_mod is responsible for exchanging 
        !! fluxes between the atmosphere, land, and ice components.
        !! @parblock
        !! Module atm_land_ice_flux_exchange_mod contains this and that
        !! @endparblock
        !! module atm_land_ice_flux_exchange
        !! end module atm_land_ice_flux_exchange
        """
        super().__init__(xmldir, xmlfile)
        self.overview = self.document_overview()
  
    def document_overview(self):
        
        briefdescription = self.get_tag("briefdescription")
        detaileddescription = self.get_tag("parblock")
        
        return f"{briefdescription}  {detaileddescription}".strip()
    

class ModuleBodyDocument(XMLsoup):

    def __init__(self,
                 xmldir: str|Path = "./docs/xml",
                 xmlfile: str|Path = None,
                 append_overview: bool = True):

        super().__init__(xmldir, xmlfile=xmlfile)
        self.bodyxmlfile = xmlfile
        self.append_overview = append_overview
        self.overview = ""
        self.variables_md = []
        self.procedures_md = []
        
        self.mdfile = [f"# {self.toplevel_name}\n"]

        if self.append_overview: 
            overviewfile = xmlfile.replace("namespace","").replace("__mod.xml", "_8_f90.xml")
            self.overview = ModuleTopLevelDocument(xmldir, overviewfile).overview
            if self.overview:
                if self.overview[-1] != ".":
                    self.overview += "."
                self.mdfile.append(f"{self.overview}\n")
            else:
                print(f"Warning: overview is empty for {self.toplevel_name} in {overviewfile}.")
                self.append_overview = False


    def document_module_variables(self):
        """
        Documents module variables.  For example, parses

          module this module
            real(8) :: var1 !< is a variable
            integer :: var2 !< is another variable
          end module this module
        
        and returns               
          ## this_module variables
          | Name | Type | Definition |
          |------|------|------------|
          | var1 | real(8) | is a variable |
          | var2 | integer | is another variable |
        All variable descriptions are assumed to be less than ~400 tokens.
        """

        variables_obj = self.soup.find_all("memberdef", {"kind": "variable"})

        if not variables_obj:
            return "There are no module variables in this module.  "

        self.variables_md.append(f"## {self.toplevel_name} variables\n")
        self.variables_md.append("| Name | Type | Definition |\n|------|------|------------|")
        for variable in variables_obj:
            varname = self.get_name(variable)
            vartype = self.get_tag_to_string("type", variable)
            briefdescription = self.get_tag_to_string("briefdescription", variable)
            self.variables_md.append(f"| {varname} | {vartype} | {briefdescription} |")

        self.variables_md.append("\n")
        self.mdfile.extend(self.variables_md)
                               
    def document_procedures(self):
        """
        Documents procedures. 
        """

        procedures_obj = self.soup.find_all("memberdef", {"kind": "function"})
        if not procedures_obj:
            return "There are no procedures in this module"

        for procedure in procedures_obj:
            
            procname = self.get_name(procedure) 
            proctype = self.get_tag_to_string("type", procedure).split(",")[0].strip() #subroutine or function
            argsstring = self.get_tag_to_string("argsstring", procedure)
            parameters_description = self.get_parameters_description(procedure, subroutine_name=procname)
            briefdescription = self.get_tag_to_string("briefdescription", procedure)
            detaileddescription = self.get_tag_to_string("parblock", procedure)            
            inbodydescription = self.get_inbodydescription(procedure)

            if briefdescription and briefdescription[-1] != ".":
                briefdescription += "."
            if detaileddescription and detaileddescription[-1] != ".":
                detaileddescription += "."

            markdown  = f"## {proctype}::{procname}\n"
            markdown += f"### intro\n"
            if self.append_overview:
                markdown += f"{self.overview}  "
            markdown += f"{procname} is a {proctype} in {self.toplevel_name}.\n"
            markdown += f"### description\n"
            markdown += f"{briefdescription}  {detaileddescription}\n"
            markdown += f"### Arguments for {procname}:\n{parameters_description}\n"
            markdown += f"### flowchart\n"
            markdown += f"{procname} does the following:  \n{inbodydescription}\n"
            self.procedures_md.append(markdown)

        self.mdfile.extend(self.procedures_md)
    
    def write_markdown(self):
        # Keep output filename path-safe while preserving module identity.
        module_name = self.toplevel_name.replace("::", "__").replace("/", "_")
        output_file = f"{module_name}.md"

        markdown_content = "\n".join(str(section) for section in self.mdfile)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        return output_file


def test():                   
    modxml = ModuleBodyDocument(xmlfile="namespaceatm__land__ice__flux__exchange__mod.xml", append_overview=True)
    modxml.document_module_variables()
    modxml.document_procedures()
    modxml.write_markdown()

if __name__ == "__main__":
    test()