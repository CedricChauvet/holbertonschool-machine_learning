#!/usr/bin/env python3
"""
task31 using mangodb, insert  a new document in a collection based on kwargs
"""
import pymongo
import pprint


def insert_school(mongo_collection, name, address):
    """ inserts a new document in a collection based on kwargs """
    mongo_collection.insert({"name": name, "address": address})
    