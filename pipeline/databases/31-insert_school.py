#!/usr/bin/env python3
"""
task31 using mangodb, insert  a new document in a collection based on kwargs
"""
import pymongo
import pprint


def insert_school(mongo_collection, **kwargs):
    """ inserts a new document in a collection based on kwargs """
    mongo_collection.insert_one(kwargs)
    