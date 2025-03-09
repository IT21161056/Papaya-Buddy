from pydantic import BaseModel

class Remedy(BaseModel):
    name: str
    desc: str