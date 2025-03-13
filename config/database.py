from pymongo import MongoClient

client = MongoClient("mongodb+srv://admin:admin123@cluster0.jxoyh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

db = client.papaya_buddy_db

# Define collections separately
papaya_collection = db["papaya_collection"]
disease_collection = db["disease_collection"]