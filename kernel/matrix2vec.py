
def vec(X):
    length = X.shape[0]
    vec = []
    for i in range(length-1):
        vec += list(X[i, i+1:])

    return vec
