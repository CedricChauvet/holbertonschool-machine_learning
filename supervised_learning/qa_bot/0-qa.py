#!/usr/bin/env python3
"""
task QA bots
using unbuntu 20.04
by Ced
"""
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    question is a string containing the question to answer
    reference is a string containing the reference document
        from which to find the answer
    Returns: a string containing the answer
    """
    # max length of the input
    max_len = 512
    # Load the tokenizer from BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    # Load the model from tensorflow hub
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # encode the inputs, question and reference
    encoded = tokenizer.encode_plus(
        question,
        reference,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors='tf'   # tensorflow in output
    )

    # Extrayez input_ids et attention_mask
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    # Créez token_type_ids
    token_type_ids = encoded['token_type_ids']

    # # print the shape and values of the inputs
    # print("input", input_ids.shape, "valeurs", input_ids)
    # print("attention", attention_mask.shape, "valeurs", attention_mask)
    # print("training", token_type_ids.shape, "valeurs", token_type_ids)

    inputs = [
        input_ids,
        attention_mask,
        token_type_ids
    ]
    result = model(inputs)

    # # Inspecter le résultat
    # print("Type de result:", type(result))
    # print("\nNombre d'éléments dans result:", len(result))

    # print("\nPour le premier élément (start_scores):")
    # print("Type:", type(result[0]))
    # print("Shape:", result[0].shape)

    # print("\nPour le deuxième élément (end_scores):")
    # print("Type:", type(result[1]))
    # print("Shape:", result[1].shape)

    # Find the tokens with the highest start and end scores

    start_index = int(np.argmax(result[0][0][1:]))
    end_index = int(np.argmax(result[1][0][1:]))
    answer_tokens = input_ids[0][start_index+1: end_index+2]
    answer = tokenizer.decode(answer_tokens)

    return answer
