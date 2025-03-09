def individual_serial(papaya) -> dict:
    return {
        "id": str(papaya["_id"]),
        "species": papaya["species"],
        "weight": papaya["weight"],
        "disease": papaya["disease"]
    }

def list_serial(papayas) -> list:
    return [individual_serial(papaya) for papaya in papayas]