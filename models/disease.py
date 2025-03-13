from typing import Optional
from pydantic import BaseModel, Field
from bson import ObjectId

class Disease(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    name: str
    affected_part: str
    symptoms: str
    disease_type: str
    description: str

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str
        }