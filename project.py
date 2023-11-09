#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import numpy as np


# In[19]:


class PhraseSimilarityCalculator:
    def __init__(self, word2vec_model_path, phrases_csv_path):
        self.word_vectors = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
        self.phrases_df = pd.read_csv(phrases_csv_path, encoding='unicode_escape')

    def calculate_batch_distances(self, distance_metric='cosine'):
        distances = np.zeros((len(self.phrases_df), len(self.phrases_df)))

        for i, phrase1 in enumerate(self.phrases_df['Phrases']):
            tokens1 = [word for word in phrase1.split() if word in self.word_vectors]

            if not tokens1:
                print(f"Skipping empty phrase: {phrase1}")
                continue

            vector1 = self.word_vectors[tokens1].mean(axis=0)

            for j, phrase2 in enumerate(self.phrases_df['Phrases']):
                tokens2 = [word for word in phrase2.split() if word in self.word_vectors]

                if not tokens2:
                    print(f"Skipping empty phrase: {phrase2}")
                    continue

                vector2 = self.word_vectors[tokens2].mean(axis=0)

                if distance_metric == 'cosine':
                    distance = cosine_distances([vector1], [vector2])[0, 0]
                elif distance_metric == 'euclidean':
                    distance = euclidean_distances([vector1], [vector2])[0, 0]
                else:
                    raise ValueError("Invalid distance metric. Use 'cosine' or 'euclidean'.")

                distances[i, j] = distance

        return distances

    def find_closest_match(self, input_phrase, distance_metric='cosine'):
        tokens_input = [word for word in input_phrase.split() if word in self.word_vectors]

        if not tokens_input:
            print("Input phrase contains words not present in the Word2Vec model.")
            return None

        vector_input = self.word_vectors[tokens_input].mean(axis=0)

        distances = []
        for phrase in self.phrases_df['Phrases']:
            tokens = [word for word in phrase.split() if word in self.word_vectors]

            if not tokens:
                print(f"Skipping empty phrase: {phrase}")
                continue

            vector = self.word_vectors[tokens].mean(axis=0)

            if distance_metric == 'cosine':
                distance = cosine_distances([vector_input], [vector])[0, 0]
            elif distance_metric == 'euclidean':
                distance = euclidean_distances([vector_input], [vector])[0, 0]
            else:
                raise ValueError("Invalid distance metric. Use 'cosine' or 'euclidean'.")

            distances.append(distance)

        closest_index = np.argmin(distances)
        closest_phrase = self.phrases_df['Phrases'][closest_index]

        return closest_phrase, distances[closest_index]


# In[20]:


# Example Usage for Batch Execution
word2vec_model_path = 'GoogleNews-vectors-negative300.bin'
phrases_csv_path = 'phrases.csv'


# In[21]:


calculator = PhraseSimilarityCalculator(word2vec_model_path, phrases_csv_path)
cosine_distances_matrix = calculator.calculate_batch_distances(distance_metric='cosine')
euclidean_distances_matrix = calculator.calculate_batch_distances(distance_metric='euclidean')


# In[22]:


# Example Usage for On-the-Fly Execution
def main():
    while True:
        user_input = input("Enter a phrase (type 'exit' to quit): ")
        
        if user_input.lower() == 'exit':
            break

        closest_match, distance = calculator.find_closest_match(user_input, distance_metric='cosine')
        
        if closest_match:
            print(f"Closest Match: {closest_match} | Distance: {distance}")
        else:
            print("No match found.")

if __name__ == '__main__':
    main()




