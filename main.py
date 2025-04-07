import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

knn_data = pd.read_csv('knn_data.csv')
temp = pd.read_csv('temp.csv')
temp = temp.drop(columns='Cluster_4')
new_input = pd.Series({
    'O_score': 7.0,
    'C_score': 6.0,
    'E_score': 7.5,
    'A_score': 6.5,
    'N_score': 5.5,
    'Numerical Aptitude': 4.0,
    'Spatial Aptitude': 4.5,
    'Perceptual Aptitude': 5.0,
    'Abstract Reasoning': 4.0,
    'Verbal Reasoning': 8.0
})

input_vector = new_input.values.reshape(1, -1)
centroid_vectors = temp.values

similarities = cosine_similarity(input_vector, centroid_vectors)

cluster = np.argmax(similarities)

print(cluster)
print(knn_data[knn_data['Cluster_4'] == cluster]['Career'].unique())