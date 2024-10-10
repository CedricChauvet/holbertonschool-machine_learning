#!/usr/bin/env python3
"""
task QA bots
using unbuntu 20.04
by Ced
"""
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

from transformers import BertTokenizer

def question_answer(question, reference):
    """
    question is a string containing the question to answer
    reference is a string containing the reference document from which to find the answer
    Returns: a string containing the answer
    """
    
    # model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
    # a) Get predictions
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

    
    QA_input = {
        'question': question,
        'context': reference
    }
    res = nlp(QA_input)

    print("Question:", QA_input['question'])
    print("Contexte:", QA_input['context'])
    print("Réponse:", res['answer'])
    print("Score de confiance:", res['score'])
