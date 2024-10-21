#!/usr/bin/env python3
"""
task QA bots
using unbuntu 20.04
by Ced
"""

# this script quit when saying bye or exit
bye = False
while not bye:
    question = input("Q: ")
    if question.lower() in ['exit', 'quit', 'goodbye', 'bye']:
        print("A: Goodbye")
        bye = True
    else:
        print("A:")
