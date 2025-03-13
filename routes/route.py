from fastapi import APIRouter
from models.papaya import Papaya
from config.database import papaya_collection
from schema.schemas import list_serial
from bson import ObjectId

router = APIRouter()

@router.get("/")
async def get_all_papayas():
    papayas = list_serial(papaya_collection.find())
    return papayas

@router.post("/create")
async def create_papaya(papaya: Papaya):
    papaya_collection.insert_one(dict(papaya))

@router.put("/{id}")
async def update_papaya(id: str, papaya: Papaya):
    papaya_collection.update_one({"_id": ObjectId(id)}, {"$set": dict(papaya)})


@router.delete("/{id}")
async def delete_papaya(id: str):
    papaya_collection.delete_one({"_id": ObjectId(id)})