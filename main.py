import gensim
import nltk
import tensorflow as tf
import numpy as np 
import pandas as pd 
import sklearn 
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
import scipy.spatial.distance

def readfile(filename):
    file_object = open(filename,'r')
    file_list = file_object.readlines()
    return file_list


def create_sentence_embedding(model,input_sentence):

    sentence_embedding = np.ones(300,)
    input_sentence = input_sentence
    word_vectors = model.wv 

    for word in input_sentence :
        if word in word_vectors.vocab : 
            vec = model[word]
            sentence_embedding = sentence_embedding * vec 

    return sentence_embedding     
    
def hierarchicalClustering(vector_of_sentences,L):
    cosine_distances = pdist(vector_of_sentences, 'cosine')
    square_form_distance = scipy.spatial.distance.squareform(cosine_distances)
    
    
    for method in ['single', 'complete', 'average', 'weighted']:
        Z = scipy.cluster.hierarchy.linkage(square_form_distance, method=method)
        R = scipy.cluster.hierarchy.inconsistent(Z, d=2)
        print(R)
        plt.clf()
        fig = plt.figure()
        ax = fig.add_axes([.1, .1, .8, .8])
        dd = scipy.cluster.hierarchy.dendrogram(Z, labels=L, leaf_font_size=7, ax=ax)
        plt.savefig('{}.pdf'.format(method))

def main(model) :
    
    sentences = readfile('StackOverflow(2).txt')
    num_of_sentences = len(sentences)
    vector_of_sentences = list()

    for i in range(num_of_sentences):
        vector_of_sentences.append(create_sentence_embedding(model,sentences[i]) )

    vector_of_sentences = np.array(vector_of_sentences)
    print (vector_of_sentences.shape)
    hierarchicalClustering(vector_of_sentences,L=list(map(str,range(num_of_sentences))))

    
        

if __name__ == '__main__' :
    model=gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz',binary=True)
    main(model)






