from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import uuid
from datetime import datetime

app = FastAPI(title="Orders Service")

# Simple in-memory database for demo
orders_db = {}

class OrderItem(BaseModel):
    product_id: str
    quantity: int
    price: float

class OrderCreate(BaseModel):
    user_id: str
    items: List[OrderItem]

class Order(BaseModel):
    id: str
    user_id: str
    items: List[OrderItem]
    total_amount: float
    status: str
    created_at: datetime

@app.get("/")
def read_root():
    return {"message": "Welcome to Orders Service"}

@app.post("/orders/", response_model=Order)
def create_order(order: OrderCreate):
    order_id = str(uuid.uuid4())
    
    # Calculate total amount
    total_amount = sum(item.price * item.quantity for item in order.items)
    
    new_order = Order(
        id=order_id,
        user_id=order.user_id,
        items=order.items,
        total_amount=total_amount,
        status="pending",
        created_at=datetime.now()
    )
    
    orders_db[order_id] = new_order
    return new_order

@app.get("/orders/", response_model=List[Order])
def read_orders():
    return list(orders_db.values())

@app.get("/orders/{order_id}", response_model=Order)
def read_order(order_id: str):
    if order_id not in orders_db:
        raise HTTPException(status_code=404, detail="Order not found")
    return orders_db[order_id]

@app.get("/users/{user_id}/orders", response_model=List[Order])
def read_user_orders(user_id: str):
    user_orders = [order for order in orders_db.values() if order.user_id == user_id]
    return user_orders

@app.put("/orders/{order_id}/status")
def update_order_status(order_id: str, status: str):
    if order_id not in orders_db:
        raise HTTPException(status_code=404, detail="Order not found")
    
    valid_statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of {valid_statuses}")
    
    orders_db[order_id].status = status
    return {"message": f"Order status updated to {status}"}

@app.delete("/orders/{order_id}")
def delete_order(order_id: str):
    if order_id not in orders_db:
        raise HTTPException(status_code=404, detail="Order not found")
    
    del orders_db[order_id]
    return {"message": "Order deleted successfully"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}