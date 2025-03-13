from typing import List, Dict
from bson import ObjectId

def disease_helper(disease: Dict) -> Dict:
    return {
        "id": str(disease.get("_id")) if disease.get("_id") else None,
        "name": disease.get("name"),
        "affected_part": disease.get("affected_part"),
        "symptoms": disease.get("symptoms"),
        "description": disease.get("description"),
        "disease_type": disease.get("disease_type")
    }

def list_disease_serial(diseases: List[Dict]) -> List[Dict]:
    return [disease_helper(disease) for disease in diseases]