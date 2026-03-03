from bs4 import BeautifulSoup

import chromadb

# read and parse xml file
xmlfile = "/home/Mikyung.Lee/fmscoupler/FMSCoupler/docs/xml/namespaceatm__land__ice__flux__exchange__mod.xml"
with open(xmlfile, "r") as openedfile:
  xmlsoup = BeautifulSoup(openedfile, "lxml-xml")
subroutines = xmlsoup.find_all("memberdef", {"kind": "function"})
variables = xmlsoup.find_all("memberdef", {"kind": "variable"})

# initialize database, use default embedding function
db_path = "./atm-land-ice-flux-exchange-human"
collection_name = "atm-land-ice-flux-exchange-human"
collection = chromadb.PersistentClient(path=db_path).get_or_create_collection(
  name=collection_name,
  metadata = {
    "description": "variables and subroutines from " + collection_name,
    "file": collection_name + ".f90"
  }
)

for variable in variables:

  name = variable.find("name").get_text(strip=True)
  vartype = variable.find("type").get_text(strip=True)
  definition = variable.briefdescription.get_text(strip=True)  
  document = f"{name} is a {vartype} module-level variable defined as the {definition}"
  
  print(document)
  collection.upsert(ids=[name], documents=[document], metadatas=[{"type":"variable"}])

for subroutine in subroutines:

  name = subroutine.find("name").get_text(strip=True)
  subtype = subroutine.find("type").get_text(strip=True)
  procedure = subtype.split(",")[0]
  argstring = subroutine.argsstring.get_text(strip=True)
  description = subroutine.briefdescription.get_text(strip=True)
  inbodydescription = subroutine.inbodydescription.get_text(strip=True)

  detaileddescription = subroutine.detaileddescription

  for child in detaileddescription.children:
    if child.name == "para":
      if list(child.children)[0].name == "parameterlist": break
      description += child.get_text()
    
  params = []
  for param in detaileddescription.find_all("parameteritem"):
    argname = param.parametername.get_text(strip=True)
    inout = param.parametername["direction"]
    definition = param.parameterdescription.get_text(strip=True)
    params.append(f"{argname} is a intent({inout}) variable defined as the {definition}")
    
  document = f"""
  {name} is a {subtype} with the following description:  {description}.
  The {procedure} has the following arguments: {params}.  
  The following occurs in the {procedure}:  {inbodydescription}
  """  

  print(document)
  collection.upsert(ids=[name], documents=[document], metadatas=[{"type":"subroutine"}])


