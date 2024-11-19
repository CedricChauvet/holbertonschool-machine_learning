#!/usr/bin/env python3
"""
Pandas project
By Ced
"""
import pandas as pd


D = dict()
D['First'] = [0.0, 0.5, 1.0, 1.5]
D['Second'] = ['one', 'two', 'three', 'four']

data = pd.DataFrame(D, index=['A', 'B', 'C', 'D'])

df = data
