#!/usr/bin/env python3
"""
task QA bots
using unbuntu 20.04
by Ced
"""
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def semantic_search(corpus, question):
    """
    performs semantic search on a corpus of documents
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')

    documents = []
    filenames = []
    for filename in os.listdir(corpus):
        if filename.endswith('.md'):
            with open(os.path.join(corpus, filename),
                      'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(content)
                filenames.append(filename)

    # Encoder la question et les documents
    doc_embeddings = model.encode(documents)
    question_embedding = model.encode([question])
    # Calculer les similarités
    similarities = cosine_similarity(question_embedding, doc_embeddings)[0]

    # Trier les résultats
    ranked_results = sorted(enumerate(similarities),
                            key=lambda x: x[1], reverse=True)
    idx, score = ranked_results[0]

    return documents[idx]
