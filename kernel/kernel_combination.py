import numpy as np


def drug_matrix_combination(fingerprint_kernel, atc_kernel, interaction_kernel, target_protein_kernel, side_effects_kernel):
    m_kernel = (fingerprint_kernel + atc_kernel + interaction_kernel)/3
    s_kernel = (target_protein_kernel + side_effects_kernel)/2

    kernel = np.zeros((fingerprint_kernel.shape[0], fingerprint_kernel.shape[0]))

    length = fingerprint_kernel.shape[0]
    for i in range(length):
        for j in range(i, length):
            if m_kernel[i, j] == 0:
                kernel[i, j] = s_kernel[i, j]
                kernel[j, i] = kernel[i, j]
            else:
                kernel[i, j] = m_kernel[i, j]
                kernel[j, i] = kernel[i, j]
    return kernel


def atc_matrix_combination(kernel_list):
    shape = kernel_list[0].shape[0]
    kernel = np.zeros((shape, shape))
    for small_kernel in kernel_list:
        kernel += small_kernel

    return kernel / len(kernel_list)
