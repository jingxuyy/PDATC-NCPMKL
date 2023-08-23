import numpy as np


def knormalized(X_matrix):
    F_start = np.dot(X_matrix, X_matrix.T)
    diag = np.diagonal(F_start)

    kernel = np.zeros((F_start.shape[0], F_start.shape[1]))

    for i in range(F_start.shape[0]):
        for j in range(F_start.shape[1]):
            if diag[i] == 0 or diag[j] == 0:
                kernel[i][j] = 0
            else:
                kernel[i][j] = F_start[i][j] / np.sqrt(diag[i] * diag[j])

    return kernel

