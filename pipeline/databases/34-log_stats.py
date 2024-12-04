#!/usr/bin/env python3
"""
task 34 :  Log stats

Write a Python script that provides some stats about Nginx logs stored in MongoDB:

Database: logs
Collection: nginx
Display (same as the example):
first line: x logs where x is the number of documents in this collection
second line: Methods:
5 lines with the number of documents with the method = ["GET", "POST", "PUT", "PATCH", "DELETE"] in this order (see example below - warning: it’s a tabulation before each line)
one line with the number of documents with:
method=GET
path=/status
"""
import pymongo
from pymongo import MongoClient
from pprint import pprint



if __name__ == "__main__":
    print("hello world")
    client = MongoClient('mongodb://127.0.0.1:27017')
    db = client["logs"]
    collection = db["nginx"]

    total_logs = collection.count_documents({})
    print(f"{total_logs} logs")
    print("Methods:")
    print("method GET:", len(list(collection.find({"method": "GET"}))))
    print("method POST:", len(list(collection.find({"method": "POST"}))))
    print("method PUT:", len(list(collection.find({"method": "PUT"}))))
    print("method PATCH:", len(list(collection.find({"method": "PATCH"}))))
    print("method DELETE:", len(list(collection.find({"method": "DELETE"}))))

