__author__ = 'xuqiongkai'

import numpy as np
import os

class Glove():
    def __init__(self, file_path):
        """
        load model(no-binary model)
        format:
        word dim1 dim2 dim3 ... dimn
        """
        with open(file_path) as f:
            self.word_dic = {line.split()[0]:np.asarray(line.split()[1:], dtype='float') for line in f}
        self.size = len(self.word_dic)
        self.dim = self.word_dic.itervalues().next().shape[0]

    def get_vector(self, word):
        return self.word_dic.get(word)

    def consine_distance(self, word1, word2):
        return np.dot(self.word_dic[word1],self.word_dic[word2]) \
            /(np.linalg.norm(self.word_dic[word1])* np.linalg.norm(self.word_dic[word2]))

    def top_k_similar_words(self, word, TopN = 10):
        return sorted({word2:self.consine_distance(word, word2) for word2 in self.word_dic.keys()}.items(), \
            lambda x, y: cmp(x[1], y[1]), reverse= True) [1:TopN+1]



if __name__ == '__main__':
    model = Glove("Models/GloveVec/vectors.6B.100d.small.txt") #load model
    #print model.top_k_similar_words('china')
    #print model.get_vector('china')
    #print model.get_vector('fdsajl;fjwaop')
