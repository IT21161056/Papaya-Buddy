from pymongo import MongoClient

client = MongoClient("mongodb+srv://admin:admin123@cluster0.jxoyh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

db = client.papaya_buddy_db

collection_name = db["papaya_collection"]