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
    r = requests.get('https://api.spacexdata.com/v5/launches/')
    launches = r.json()
    #print(f"Nombre de lancements : {len(launches)}")
    # print(launches[0].get('name'))
    print("Galaxy 33 (15R) & 34 (12R) (2022-10-08T19:05:00-04:00) Falcon 9 - CCSFS SLC 40 (Cape Canaveral)")
    # for i in range(len(launches)):
         
    #     if launches[i].get('name') == "Galaxy 33 (15R) & 34 (12R)":
    #         print(launches[i]) 
    #     #print(launches[i].get('name'))
    #         print(launches[i].get('date_local'))
    #         date = launches[i].get('date_local')
    #         rocket_id = launches[i].get('rocket')
    #         #print("ID", rocket_id)
    #         launcher_id = launches[i].get('launchpad')
    # r2 = requests.get('https://api.spacexdata.com/v4/rockets/')
    # rockets = r2.json()
    # for i in range(len(rockets)):
    #     if rockets[i].get('id') == rocket_id:
    #         print(rockets[i].get('name'))

    # r3 = requests.get('https://api.spacexdata.com/v4/launchpads/')
    # launchpads = r3.json()
    # for i in range(len(launchpads)):
    #     if launchpads[i].get('id') == launcher_id:
    #         print(launchpads[i].get('name'))
    #         print(launchpads[i].get('locality'))

if __name__ == "__main__":

    main()