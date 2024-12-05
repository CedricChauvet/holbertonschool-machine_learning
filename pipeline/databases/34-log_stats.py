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
    client = MongoClient('mongodb://127.0.0.1:27017')
    db = client["logs"]
    collection = db["nginx"]

    total_logs = collection.count_documents({})

    #print(list(collection.find({"path": "/status"}).limit(1)))


    print(f"{total_logs} logs")
    print("Methods:")
    print("\tmethod GET:", len(list(collection.find({"method": "GET"}))))
    print("\tmethod POST:", len(list(collection.find({"method": "POST"}))))
    print("\tmethod PUT:", len(list(collection.find({"method": "PUT"}))))
    print("\tmethod PATCH:", len(list(collection.find({"method": "PATCH"}))))
    print("\tmethod DELETE:", len(list(collection.find({"method": "DELETE"})))) 

    print (f"{len(list(collection.find({'method': 'GET', 'path': '/status'})))} status check")