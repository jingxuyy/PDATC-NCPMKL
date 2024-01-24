import numpy as np
from sklearn import metrics
from sklearn.metrics import pairwise_distances


class KernelMatrix(object):
    """
    Note:
    1. Construct the kernel matrix through the original matrix. This class only supports construction by row (index=0)
        and construction by column (index=1).
    2. This class realizes the construction of five kernel matrices, which are:
        Gaussian kernel kernel2gip, correlation coefficient kernel kernel2corr, cosine kernel kernel2cos,
        mutual information kernel kernel2mi, jaccard similarity kernel kernel2jaccard.
    3. The definition of its Kernel function can be seen in the papers
        "Identification of Drug-side Effect Association via Semi-supervised Model and Multiple Kernel Learning"
        and "A novel approach based on deep residual learning to predict drug's anatomical therapeutic chemical code ".
    4. Note that you can only determine the direction of the constructor kernel each time you declare an object,
        so if you want to construct a row kernel and a column kernel, you need to instantiate both object constructs.
    5. In the construction of the kernel function, attention should be paid to the case that a certain column or a certain behavior is all 0.
        In this case, the construction of the kernel function will make mistakes,
        for example, in the construction of the correlation coefficient kernel kernel2corr and cosine kernel kernel2cos.
        It is necessary to add relatively small Gaussian noise to the original matrix.
    """
    def __init__(self, data, index=0):
        self.data = data
        self.index = index

    def transpose(self):
        # X = self.data.values
        X = self.data
        if self.index == 0:
            return X
        elif self.index == 1:
            return X.T
        else:
            print('The value of index can only be 0 or 1')
            return X

    def kernel2corr(self, noise):
        X = self.transpose() + noise
        return np.corrcoef(X)

    def kernel2cos(self, noise):
        X = self.transpose() + noise
        num = np.dot(X, X.T)  
        a = np.linalg.norm(X, axis=1).reshape(1, -1)
        b = np.linalg.norm(X, axis=1).reshape(1, -1)
        denom = np.dot(a.T, b)

        kernel = num / denom
        return kernel

    def kernel2gip(self, gamma):
        X = self.transpose()
        length = X.shape[0]
        kernel = np.zeros((length, length))

        for i in range(length):
            temp = X - X[i, :].reshape(1, -1)
            norm = np.linalg.norm(temp, axis=1) ** 2
            kernel[i, :] = np.exp(-gamma * norm)

        return kernel

    def kernel2jaccard(self):
        X = self.transpose()
        return 1 - pairwise_distances(X, metric='jaccard')

    def kernel2mi(self):
        X = self.transpose()
        length = X.shape[0]
        kernel = np.zeros((length, length))

        for i in range(length):
            a_vec = X[i, :]
            for j in range(i+1):
                b_vec = X[j, :]
                kernel[i, j] = metrics.mutual_info_score(a_vec, b_vec)
                kernel[j, i] = kernel[i, j]
        return kernel

    def __str__(self):
        print('The input matrix is:')
        print(self.data)
        if self.index == 0:
            return 'The calculated kernel matrix is calculated according to the rows of the original matrix'
        elif self.index == 1:
            return 'The calculated kernel matrix is calculated according to the columns of the original matrix'
        else:
            return 'The direction you entered is not valid'


