import numpy as np


class ReformulateAdjacencyMatrix(object):

    def __init__(self, index_similarity, column_similarity, adjacency_matrix, step):
        self.index_similarity = index_similarity
        self.column_similarity = column_similarity
        self.adjacency_matrix = adjacency_matrix
        self.step = step
        self.index_num = self.adjacency_matrix.shape[0]
        self.column_num = self.adjacency_matrix.shape[1]


    def vertical(self):
        matrix = np.zeros((self.index_num, self.column_num))
        for i in range(self.column_num):
            Wd = 0
            DA = np.zeros(self.index_num)
            sort_similarity = (-self.column_similarity[i]).argsort()
            for j in range(self.step):
                Wd += self.column_similarity[i, sort_similarity[j]]
                DA += self.column_similarity[i, sort_similarity[j]]*self.adjacency_matrix[:, sort_similarity[j]]
            matrix[:, i] = DA / Wd

        return matrix

    def horizontal(self):
        matrix = np.zeros((self.index_num, self.column_num))
        for i in range(self.index_num):
            Wc = 0
            CA = np.zeros(self.column_num)
            sort_similarity = (-self.index_similarity[i]).argsort()
            for j in range(self.step):
                Wc += self.index_similarity[i, sort_similarity[j]]
                CA += self.index_similarity[i, sort_similarity[j]]*self.adjacency_matrix[sort_similarity[j], :]
            matrix[i, :] = CA / Wc
        return matrix


    def final_reformulation(self):
        sum_matrix = (np.nan_to_num(self.horizontal()) + np.nan_to_num(self.vertical()))/2

        return np.where((sum_matrix+self.adjacency_matrix) > 1, 1, sum_matrix)







