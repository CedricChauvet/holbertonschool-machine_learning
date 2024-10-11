#!/usr/bin/env python3
"""
task QA bots
using unbuntu 20.04
by Ced
"""

question_answer = __import__('0-qa').question_answer
def answer_loop(reference):
    """
    reference is the string containing the reference document from which to find the answer
    If the answer cannot be found in the reference document, respond with Sorry, I do not understand your question.
    """
    bye = False
    while not bye:
        question = input("Q: ")
        if question.lower() in ['exit', 'quit', 'goodbye', 'bye']:
            print("A: Goodbye")
            bye = True
        else:
            print("A:", question_answer(question, reference))
