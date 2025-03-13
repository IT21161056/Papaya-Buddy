from pydantic import BaseModel, Field
from bson import ObjectId
from typing import Optional, List
from datetime import datetime

class History(BaseModel):
    id: Optional[str] = Field(alias="_id")
    user_uploaded_img_url: str
    userid: str
    Treatment: List[ObjectId]
    Disease: ObjectId
    suggested_images: List[ObjectId]
    created_at: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat() if isinstance(v, datetime) else None
        }