import ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def open_vectors_file(vector_file):
    vector_dict = {}
    with open(vector_file, encoding="utf8") as file:
        for line in file:
            line = line.strip("\n")
            norms = line.split(" ")[0]
            vectors = line.split(" ", 1)[1]
            vectors = ast.literal_eval(vectors)
            vector_dict[norms] = vectors        
    return vector_dict
    
vector_file = "asc_sorted_norm_food_vectors.txt"

def tsne_plot(vector_dict):
    # Creates and TSNE model and plots it
    norms = []
    vectors = []
    vector_dict = open_vectors_file(vector_file)
    for norm, vector in vector_dict.items():
        norms.append(norm)
        vectors.append(vector)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(vectors)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(norms[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

tsne_plot(open_vectors_file(vector_file))