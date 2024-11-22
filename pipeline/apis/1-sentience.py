#!/usr/bin/env python3
"""
APIs project
By Ced
"""
import requests


def sentientPlanets():
    listPlanets = []
    for i in range(70):
        r = requests.get('https://swapi-api.hbtn.io/api/species/' + str(i))
        specie_class = r.json().get('classification')
        specie_designation = r.json().get('designation')
        # print(specie_designation)
        if specie_class == 'sentient' or specie_designation == 'sentient':
            
            planet_url = r.json().get('homeworld')
            # print("url", planet_url)
            if planet_url is not None:
                planet = requests.get(planet_url).json().get('name')
                listPlanets.append(planet)

    return listPlanets