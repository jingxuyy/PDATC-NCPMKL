import numpy as np


class NSP(object):
    def __init__(self, drug_similarity, atc_similarity, adjacency_matrix):
        self.drug_similarity = drug_similarity
        self.atc_similarity = atc_similarity
        self.adjacency_matrix = adjacency_matrix


    def ATCSP(self):
        temp_matrix = np.dot(self.adjacency_matrix, self.atc_similarity)
        modulus = np.linalg.norm(self.adjacency_matrix, axis=1).reshape(-1, 1)
        return temp_matrix / modulus

    def DrugSP(self):
        temp_matrix = np.dot(self.drug_similarity, self.adjacency_matrix)
        modulus = np.linalg.norm(self.adjacency_matrix, axis=0).reshape(1, -1)
        return temp_matrix / modulus

    def calculate_modulus_sum(self):
        index_modulus = np.linalg.norm(self.drug_similarity, axis=1).reshape(-1, 1)
        columns_modulus = np.linalg.norm(self.atc_similarity, axis=0).reshape(1, -1)
        return index_modulus + columns_modulus

    def network_NSP(self):
        result = np.nan_to_num((np.nan_to_num(self.ATCSP())+np.nan_to_num(self.DrugSP()))/self.calculate_modulus_sum())
        return result

