import math
import numpy as np


class Cross_validation(object):
    def __init__(self, data):
        self.data = data
        self.length = len(self.data)

    def K_fold(self, base_num):
        split_list = []
        min_part = math.floor(self.length / base_num)

        max_index = 0
        for i in range(base_num):
            temp_data = self.data.copy()
            if i == base_num - 1:
                temp_data.iloc[max_index:, :] = 0
                test_index = [j for j in range(max_index, self.length)]
            else:
                temp_data.iloc[max_index: (i + 1) * min_part, :] = 0
                test_index = [j for j in range(max_index, (i + 1) * min_part)]
            max_index = (i + 1) * min_part

            split_list.append((test_index, temp_data))
        return split_list

    def sample_loc(self):
        positive_index, positive_column = np.where(self.data == 1)
        negative_index, negative_column = np.where(self.data == 0)

        positive_matrix_loc = np.array([positive_index, positive_column]).T
        negative_matrix_loc = np.array([negative_index, negative_column]).T

        np.random.shuffle(positive_matrix_loc)
        np.random.shuffle(negative_matrix_loc)

        return positive_matrix_loc, negative_matrix_loc

    def one2one_K_fold(self, base_num):
        positive_matrix_loc, negative_matrix_loc = self.sample_loc()
        split_list = []
        positive_length = len(positive_matrix_loc)
        min_part = math.floor(positive_length / base_num)

        max_index = 0
        for i in range(base_num):
            temp_data = self.data.copy()
            if i == base_num - 1:
                temp_positive_matrix_loc = positive_matrix_loc[max_index:, :]
                temp_negative_matrix_loc = negative_matrix_loc[max_index:max_index + len(temp_positive_matrix_loc), :]

                m_index = temp_positive_matrix_loc[:, 0]
                m_column = temp_positive_matrix_loc[:, 1]

                temp_data.values[m_index, m_column] = 0

            else:
                temp_positive_matrix_loc = positive_matrix_loc[max_index: (i + 1) * min_part, :]
                temp_negative_matrix_loc = negative_matrix_loc[max_index: (i + 1) * min_part, :]

                m_index = temp_positive_matrix_loc[:, 0]
                m_column = temp_positive_matrix_loc[:, 1]

                temp_data.values[m_index, m_column] = 0

            max_index = (i + 1) * min_part

            split_list.append((temp_positive_matrix_loc, temp_negative_matrix_loc, temp_data))

        return split_list


