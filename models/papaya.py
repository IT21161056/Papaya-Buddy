from pydantic import BaseModel

class Papaya(BaseModel):
    species: str
    weight: str
    disease: str