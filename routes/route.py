from fastapi import APIRouter
from models.papaya import Papaya
from config.database import collection_name
from schema.schemas import list_serial
from bson import ObjectId

router = APIRouter()

@router.get("/")
async def get_all_papayas():
    papayas = list_serial(collection_name.find())
    return papayas

@router.post("/create")
async def create_papaya(papaya: Papaya):
    collection_name.insert_one(dict(papaya))

@router.put("/{id}")
async def update_papaya(id: str, papaya: Papaya):
    collection_name.update_one({"_id": ObjectId(id)}, {"$set": dict(papaya)})


@router.delete("/{id}")
async def delete_papaya(id: str):
    collection_name.delete_one({"_id": ObjectId(id)})