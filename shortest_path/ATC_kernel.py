import numpy as np
import pandas as pd


class Similarity(object):

    def __init__(self, data):
        self.data = data
        self.num = self.data.shape[0]
        self.length = self.data.shape[1]

    def probabilistic_kernel(self, path, gamma=0.25, level=1):

        ATC_shortest_path_length_matrix = pd.read_csv(path, index_col=0)
        label_num = list(self.data.sum(axis=0))
        scores = np.zeros((self.length, self.length))

        for i in range(self.length):
            for j in range(i + 1):
                scores[i, j] = (label_num[i] / self.num) * (label_num[j] / self.num) * np.exp(-gamma * self.get_iloc(ATC_shortest_path_length_matrix, i, j))
                scores[j, i] = scores[i, j]
        return scores

    def get_iloc(self, ATC_shortest_path_length_matrix, i, j):
        path = ATC_shortest_path_length_matrix.iloc[i, j]
        return path


class Layered(object):
    def __init__(self, ATC_list, level):
        self.ATC_list = ATC_list
        self.length = len(self.ATC_list)
        self.kernel = np.zeros((self.length, self.length))
        self.level = level
        if not self.check():
            exit(-1)

    def check(self):
        atc_str = self.ATC_list[0]
        if self.level == 1 and len(atc_str) != 1:
            print('层数和ATC代码不匹配')
            return False
        if self.level == 2 and len(atc_str) != 3:
            print('层数和ATC代码不匹配')
            return False
        if self.level == 3 and len(atc_str) != 4:
            print('层数和ATC代码不匹配')
            return False
        if self.level == 4 and len(atc_str) != 5:
            print('层数和ATC代码不匹配')
            return False
        if self.level == 5 and len(atc_str) != 7:
            print('层数和ATC代码不匹配')
            return False
        return True


    def calculate(self, x, y, n=1):
        x = set(x)
        y = set(y)

        inter_length = len(x.intersection(y))

        return (2 * inter_length + n) / (len(x) + len(y) + n)

    def ATC_split(self, atc, level):
        atc_list = []
        if level == 1:
            atc_list.append(atc)
        elif level == 2:
            atc_list.append(atc[0])
            atc_list.append(atc)
        elif level == 3:
            atc_list.append(atc[0])
            atc_list.append(atc[:3])
            atc_list.append(atc)
        elif level == 4:
            atc_list.append(atc[0])
            atc_list.append(atc[:3])
            atc_list.append(atc[:4])
            atc_list.append(atc)
        elif level == 5:
            atc_list.append(atc[0])
            atc_list.append(atc[:3])
            atc_list.append(atc[:4])
            atc_list.append(atc[:5])
            atc_list.append(atc)
        else:
            print("level参数错误")
            return []
        return atc_list

    def get_SM_kernel(self, n=1):
        for i in range(self.length):
            atc_a_list = self.ATC_split(self.ATC_list[i], self.level)
            for j in range(i+1):
                atc_b_list = self.ATC_split(self.ATC_list[j], self.level)
                self.kernel[i][j] = self.calculate(atc_a_list, atc_b_list, n)
                self.kernel[j][i] = self.kernel[i][j]
        return self.kernel





