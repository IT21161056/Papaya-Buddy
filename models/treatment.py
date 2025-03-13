from pydantic import BaseModel, Field
from bson import ObjectId
from typing import Optional
from datetime import datetime

class Treatment(BaseModel):
    id: Optional[str] = Field(alias="_id")
    method: str
    description: str
    Disease: ObjectId
    History: ObjectId

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat() if isinstance(v, datetime) else None
        }