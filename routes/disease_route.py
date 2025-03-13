from fastapi import APIRouter, HTTPException
from bson import ObjectId
from bson.errors import InvalidId
from typing import List, Dict
from models.disease import Disease
from schema.disease_helper import disease_helper, list_disease_serial
from config.database import disease_collection

disease_router = APIRouter()

@disease_router.post("/create_disease", response_model=Dict)
async def create_disease(disease: Disease) -> Dict:
    try:
        disease_data = disease.dict(exclude={"id"})
        
        disease_data["_id"] = ObjectId()

        result = disease_collection.insert_one(disease_data)

        if result.inserted_id:
            disease_data["id"] = str(result.inserted_id)
            return disease_helper(disease_data)
        else:
            raise HTTPException(status_code=500, detail="Failed to insert disease")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating disease: {str(e)}")
    
    

@disease_router.get("/disease/{id}", response_model=Dict)
async def get_disease(id: str) -> Dict:
    try:
        disease =  disease_collection.find_one({"_id": ObjectId(id)})
        if not disease:
            raise HTTPException(status_code=404, detail="Disease not found")
        return disease_helper(disease)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid ID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving disease: {str(e)}")
    
    

@disease_router.get("/disease", response_model=List[Dict])
async def get_all_diseases() -> List[Dict]:
    try:
        diseases =  disease_collection.find().to_list(None)

        return list_disease_serial(diseases)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving diseases: {str(e)}")