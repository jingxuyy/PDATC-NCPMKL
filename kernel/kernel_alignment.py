import numpy as np
from kernel.matrix2vec import vec
import cvxpy as cp


def alignment(K_ideal, K_gip, K_corr, K_cos, K_jaccard, K_mi):
    x_gip = cp.Variable(pos=True)
    x_corr = cp.Variable(pos=True)
    x_cos = cp.Variable(pos=True)
    x_jaccard = cp.Variable(pos=True)
    x_mi = cp.Variable(pos=True)

    variable = x_gip + x_corr + x_cos + x_jaccard + x_mi
    constraints = [variable == 1, x_gip >= 0, x_corr >= 0, x_cos >= 0, x_jaccard >= 0, x_mi >= 0]

    u_K_start = np.array(vec(K_ideal))

    K_gip_vec = np.array(vec(K_gip))
    K_corr_vec = np.array(vec(K_corr))
    K_cos_vec = np.array(vec(K_cos))
    K_mi_vec = np.array(vec(K_mi))
    K_jaccard_vec = np.array(vec(K_jaccard))

    u_d_start = x_gip * K_gip_vec + x_corr * K_corr_vec + x_cos * K_cos_vec + x_mi * K_mi_vec + x_jaccard * K_jaccard_vec

    objective = cp.Minimize(np.linalg.norm(u_K_start) * cp.norm(u_d_start, 2) - cp.sum(cp.multiply(u_d_start, u_K_start)))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)

    return x_gip.value * K_gip + x_corr.value * K_corr + x_cos.value * K_cos + x_jaccard.value * K_jaccard + x_mi.value * K_mi



