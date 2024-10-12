from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

def embed_document(text, max_length=512):
    # Tokenize et encode le texte
    inputs = tokenizer(text, return_tensors='tf', max_length=max_length, truncation=True, padding='max_length')
    
    # Obtenir les embeddings BERT
    outputs = model(inputs)
    
    # Utiliser l'embedding du token [CLS] comme représentation du document
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    print("cls_embedding", cls_embedding.shape)
    return cls_embedding

def semantic_search(corpus_path, query):
    query_embedding = embed_document(query)
    
    best_match = None
    best_score = -1
    
    for file in os.listdir(corpus_path):
        if file.endswith(".md"):
            with open(os.path.join(corpus_path, file), "r", encoding="utf-8") as f:
                content = f.read()
                doc_embedding = embed_document(content)
                
                similarity = cosine_similarity(query_embedding, doc_embedding)[0]
                print("file", file, "score", similarity)
                if similarity > best_score:
                    best_score = similarity
                    best_match = file
    
    return best_match, best_score