#!/usr/bin/env python3
"""
task31 using mangodb, insert  a new document in a collection based on kwargs
"""
import pymongo


def update_topics(mongo_collection, name, topics):
    """
    updates all topics of a school document based on the name
    using the following method: def update_topics(mongo_collection, name, topics):
    with pymango
    """
    myquery = { name: name }
    newvalues = { "$set": { topics: topics } }

    mongo_collection.update_one(myquery, newvalues)