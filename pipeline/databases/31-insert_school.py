#!/usr/bin/env python3
import pymongo
import pprint


def insert_school(mongo_collection, name, address):
    """ inserts a new document in a collection based on kwargs """
    mongo_collection.insert_one({"name": name, "address": address})
    