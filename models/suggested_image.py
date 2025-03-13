from pydantic import BaseModel, Field
from bson import ObjectId
from typing import Optional, List

class SuggestedImage(BaseModel):
    id: Optional[str] = Field(alias="_id")
    Disease: ObjectId
    url: List[str]

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str
        }