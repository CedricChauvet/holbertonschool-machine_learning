#!/usr/bin/env python3
"""
task QA bots
using unbuntu 20.04
by Ced
"""
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np
question_answer = __import__('0-qa').question_answer


def question_answer(corpus_path):
    """
    this fonction answer questions from a corpus of documents
    it summarize the selected document and return the answer
    """
    model_1 = SentenceTransformer('all-MiniLM-L6-v2')

    documents = []
    filenames = []
    for filename in os.listdir(corpus_path):
        if filename.endswith('.md'):
            with open(os.path.join(corpus_path, filename),
                      'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(content)
                filenames.append(filename)
    doc_embeddings = model_1.encode(documents)

    # max length of the input
    max_len = 512
    # Load the tokenizer from BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Load the model from tensorflow hub
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    bye = False
    while not bye:
        question = input("Q: ")
        question_embedding = model_1.encode([question])
        # Calculer les similarités
        similarities = cosine_similarity(question_embedding, doc_embeddings)[0]

        # Trier les résultats
        ranked_results = sorted(enumerate(similarities),
                                key=lambda x: x[1], reverse=True)
        idx, score = ranked_results[0]
        document = documents[idx]
        # print("type", type(document), document)

        if question.lower() in ['exit', 'quit', 'goodbye', 'bye']:
            print("A: Goodbye")
            bye = True
        else:
            # encode the inputs, question and reference
            encoded = tokenizer.encode_plus(
                question,
                document,
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

            inputs = [
                input_ids,
                attention_mask,
                token_type_ids
            ]
            result = model(inputs)
            start_index = int(np.argmax(result[0][0][1:]))
            end_index = int(np.argmax(result[1][0][1:]))
            answer_tokens = input_ids[0][start_index+1: end_index+2]
            answer = tokenizer.decode(answer_tokens)
            if answer is None or answer == "":
                print("A: Sorry, I do not understand your question.")
            else:
                print("A:", answer)
