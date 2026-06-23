from pydantic import BaseModel

class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""
    source: str
    name: str
    parent: str
    ichunk: int = 1
    datatype: str 
