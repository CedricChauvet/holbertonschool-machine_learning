#!/usr/bin/env python3
"""
task QA bots
using unbuntu 20.04
by Ced
"""
import os
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

def semantic_search(corpus_path, sentence):
    """
    performs semantic search on a corpus of documents
    """

    max_len = 512
    # Load the tokenizer from BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Load the model from tensorflow hub
    model = TFBertModel.from_pretrained('bert-base-uncased')
    
    # max length of the input 
    sentence_embedding = embed(sentence,model,tokenizer,max_len)
    print(f"out_sentence shape: {sentence_embedding.shape}")
    files = os.listdir(corpus_path)
    similarity_list_corpus = np.array([])
    for file in files: 
        print(f"file: {file}")
        if file.endswith(".md"):
            with open(f"./{corpus_path}/{file}", "r", encoding="utf-8") as fichier:
                doc = fichier.read()
                doc_embedding = embed(doc,model,tokenizer,max_len)

                                # Cela nous donne un vecteur unique pour chaque document
                mean_sentence_emb = tf.reduce_mean(sentence_embedding, axis=1)  # Shape: [1, hidden_size]
                mean_doc_emb = tf.reduce_mean(doc_embedding, axis=1)  # Shape: [1, hidden_size]
                
                # 2. Calculons la similarité cosinus
                # Squeeze pour enlever la dimension batch si nécessaire
                mean_sentence_emb = tf.squeeze(mean_sentence_emb)  # Shape: [hidden_size]
                mean_doc_emb = tf.squeeze(mean_doc_emb)  # Shape: [hidden_size]
                
                # Calcul du produit scalaire
                dot_product = tf.reduce_sum(mean_sentence_emb * mean_doc_emb)
                
                # Calcul des normes
                norm_sentence = tf.norm(mean_sentence_emb)
                norm_doc = tf.norm(mean_doc_emb)
                
                # Similarité cosinus
                cosine_similarity = dot_product / (norm_sentence * norm_doc)
                similarity_list_corpus = np.append(similarity_list_corpus, cosine_similarity.numpy())
                
    print(f"similarity_list_corpus: {similarity_list_corpus}")
    numero_doc = np.argmax(similarity_list_corpus)
    print(f"Best sentence in {numero_doc}")
    return None
    
def embed(text,model,tokenizer,max_len):

    # encode the inputs, question and reference
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors='tf'# tensorflow in output
    )
        
    # Extrayez input_ids et attention_mask
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    inputs = [
        input_ids,
        attention_mask,
    ]

    result = model(inputs)
    token_embeddings = result.last_hidden_state

    return token_embeddings