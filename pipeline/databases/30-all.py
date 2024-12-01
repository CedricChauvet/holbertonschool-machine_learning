#!/usr/bin/env python3
""" 30 list all school in a database """
import pymongo
import pprint


def list_all(mongo_collection):
    """ lists all documents in a collection """

    return list(mongo_collection.find())