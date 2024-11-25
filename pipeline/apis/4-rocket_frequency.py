#!/usr/bin/env python3
"""
APIs project
By Ced
"""
import requests


def main():
    """
    main function
    """

    # get the data from the API launch AND rockets
    r = requests.get('https://api.spacexdata.com/v4/launches/')
    r2 = requests.get('https://api.spacexdata.com/v4/rockets/')
    launches = r.json()
    rockets = r2.json()

    # initialize count
    falcon_1_count = 0
    falcon_9_count = 0
    falcon_heavy_count = 0

    # before we store the ids corresponding to the rockets
    for i in rockets:
        if i.get('name') == "Falcon 1":
            falcon_1_id = i.get('id')
        if i.get('name') == "Falcon 9":
            falcon_9_id = i.get('id')
        if i.get('name') == "Falcon Heavy":
            falcon_heavy_id = i.get('id')

    # we loop through the launches to count the number of
    # launches for each rocket
    for i in range(len(launches)):

        if launches[i].get('rocket') == falcon_1_id:
            falcon_1_count += 1
        if launches[i].get('rocket') == falcon_9_id:
            falcon_9_count += 1
        if launches[i].get('rocket') == falcon_heavy_id:
            falcon_heavy_count += 1

    print(f"Falcon 9: {falcon_9_count}")
    print(f"Falcon 1: {falcon_1_count}")
    print(f"Falcon Heavy: {falcon_heavy_count}")


if __name__ == "__main__":

    main()
