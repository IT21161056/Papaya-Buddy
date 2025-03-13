#create a disease
    http://localhost:5080/disease/create_disease
    {
        "name": "Papaya Mealy Bug",
        "affected_area": "Fruit",
        "symptoms": "Sooty malls on fruits",
        "description": "A viral pest affecting papaya fruits.",
        "disease_type":"Pest"
    }


#create a treatment
    http://localhost:5080/treatment/create_treatment
    {
        "method": "Chemical",
        "description": "Complete removal of the affected area.",
        "diseaseId": "67d3048b896e515daefee320",
        "historyId": "64f1b2b3c9e77b001a8e8e8f"
    }


#create suggested images
    http://localhost:5080/suggested_image/suggested_image
    {
        "diseaseId": "67d2d7524b7a26fa3094bbb4",
        "urls":["uiwuiuiuhieuihwuhiee","ewuiebbjkwenjewnjew"]
    }


#create new prediction history
    http://localhost:5080/history/create_history
    {
        "uploaded_img_url": "https://example.com/werwerewrwerewew.jpg",
        "userid": "user123",
        "treatmentId": ["67d2fe4e067c164314db28ae"],
        "diseaseId": "67d3048b896e515daefee320",
        "suggested_image_list_id": "67d305246aaffcc0e6631788"
    }
#get history by user id
    http://localhost:5080/history/get_user_history/user123