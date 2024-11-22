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

    # initialize ships list and set flag
    ships = []
    set_true = False

    for i in range(70):
        r = requests.get('https://swapi-api.hbtn.io/api/starships/' + str(i))
        passenger = r.json().get('passengers')

        if passenger is not None:
            passenger = passenger.replace(",", "")
            if passenger.isnumeric():

                if int(passenger) >= passengerCount:
                    set_true = True
                    ships.append(r.json().get('name'))

    # return [] if no ships found
    if set_true is False:
        return []

    else:
        return ships
