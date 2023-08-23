import numpy as np
import pandas as pd


class WKNKN(object):

    def __init__(self, index_similarity, column_similarity, adjacency_matrix, weight):
        self.index_similarity = index_similarity
        self.column_similarity = column_similarity
        self.adjacency_matrix = adjacency_matrix
        self.weight = weight

    def horizontal(self):
        all_scores = []
        for i in range(self.index_similarity.shape[0]):
            W = []
            sort_similarity = (-self.index_similarity[i]).argsort()
            for j in range(len(sort_similarity)):
                W.append((self.weight ** j) * self.index_similarity[i, sort_similarity[j]])
            Z_d = 1 / np.sum(self.index_similarity[i, sort_similarity])
            score = Z_d * np.dot(W, self.adjacency_matrix[sort_similarity, :])
            all_scores.append(list(score))

        return pd.DataFrame(all_scores).values

    def vertical(self):
        all_scores = []
        for i in range(self.column_similarity.shape[0]):
            W = []
            sort_similarity = (-self.column_similarity[i]).argsort()
            for j in range(len(sort_similarity)):
                W.append((self.weight ** j) * self.column_similarity[i, sort_similarity[j]])
            Z_d = 1 / np.sum(self.column_similarity[i, sort_similarity])
            score = Z_d * np.dot(self.adjacency_matrix[:, sort_similarity], np.array(W).T)
            all_scores.append(list(score))

        return pd.DataFrame(all_scores).values.T

    def get_scores(self):
        sum_matrix = (np.nan_to_num(self.horizontal()) + np.nan_to_num(self.vertical())) / 2
        return np.where((sum_matrix + self.adjacency_matrix) > 1, 1, sum_matrix)

