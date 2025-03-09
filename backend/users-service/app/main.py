from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import uuid

app = FastAPI(title="Users Service")

# Simple in-memory database for demo purposes
users_db = {}

class UserCreate(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None

class User(BaseModel):
    id: str
    username: str
    email: str
    full_name: Optional[str] = None

@app.get("/")
def read_root():
    return {"message": "Welcome to Users Service"}

@app.post("/users/", response_model=User)
def create_user(user: UserCreate):
    user_id = str(uuid.uuid4())
    new_user = User(
        id=user_id,
        username=user.username,
        email=user.email,
        full_name=user.full_name
    )
    users_db[user_id] = new_user
    return new_user

@app.get("/users/", response_model=List[User])
def read_users():
    return list(users_db.values())

@app.get("/users/{user_id}", response_model=User)
def read_user(user_id: str):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return users_db[user_id]

@app.put("/users/{user_id}", response_model=User)
def update_user(user_id: str, user: UserCreate):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    updated_user = User(
        id=user_id,
        username=user.username,
        email=user.email,
        full_name=user.full_name
    )
    users_db[user_id] = updated_user
    return updated_user

@app.delete("/users/{user_id}")
def delete_user(user_id: str):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    del users_db[user_id]
    return {"message": "User deleted successfully"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}