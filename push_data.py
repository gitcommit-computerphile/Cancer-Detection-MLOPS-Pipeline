"""
push_data.py — run once to load the breast cancer dataset into MongoDB Atlas.

Usage:
    python push_data.py
"""
import os
import sys
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.datasets import load_breast_cancer

load_dotenv()

MONGO_DB_URL = os.getenv("MONGODB_URL_KEY")
if not MONGO_DB_URL:
    raise EnvironmentError("MONGODB_URL_KEY not set in .env")

DATABASE = "cancer"
COLLECTION = "breast_cancer"


def push_data():
    data = load_breast_cancer(as_frame=True)
    df = data.frame  # 569 rows, 30 feature cols + 'target'

    client = MongoClient(MONGO_DB_URL)
    collection = client[DATABASE][COLLECTION]

    existing = collection.count_documents({})
    if existing > 0:
        print(f"Collection already has {existing} documents. Skipping.")
        print("To re-push, drop the collection in Atlas first.")
        return

    records = df.to_dict(orient="records")
    collection.insert_many(records)
    print(f"Pushed {len(records)} records to {DATABASE}.{COLLECTION}")


if __name__ == "__main__":
    push_data()