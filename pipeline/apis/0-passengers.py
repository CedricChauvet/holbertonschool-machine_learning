#!/usr/bin/env python3
"""
APIs project
By Ced
"""
import requests


def availableShips(passengerCount):
    """
    get available ships with capacity
    return list of ships
    """
    ships = []
    # rg1 = requests.get('https://swapi-api.hbtn.io/api/starships/'
    #                     ).json()
    
    # rg2 = requests.get("https://swapi-api.hbtn.io/api/starships/?page=2").json().count
    # print(rg2)

    set_true = False
    for i in range(70):
        r = requests.get('https://swapi-api.hbtn.io/api/starships/'+ str(i))
        passenger = r.json().get('passengers')
        if passenger is not None and passenger.isnumeric():
            set_true= True
        
            if int(passenger) >= passengerCount:
                # print(r.json().get('name'))
                ships.append(r.json().get('name'))
    if set_true == False:
        return []
    else: 
        return ships