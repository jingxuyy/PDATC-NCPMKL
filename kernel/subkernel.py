from kernel.kernel import KernelMatrix
import numpy as np


def generative_sub_kernel(matrix, mu=0.005, gamma1=0.002, gamma2=1):
    KL = KernelMatrix(matrix)

    noise = np.random.normal(loc=mu, scale=gamma1, size=(matrix.shape[0], matrix.shape[1]))

    K_gip = KL.kernel2gip(gamma2)
    K_corr = KL.kernel2corr(noise)
    K_cos = KL.kernel2cos(noise)
    K_jaccard = KL.kernel2jaccard()
    K_mi = KL.kernel2mi()

    return K_gip, K_corr, K_cos, K_jaccard, K_mi
