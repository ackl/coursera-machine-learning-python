def findClosestCentroids(X, centroids):
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)

    for i, x in enumerate(X):
        distance = [np.sqrt(np.sum(np.square(x - c))) for c in centroids]
        idx[i] = np.argmin(distance)
    
    return idx

def computeCentroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    Ck = np.zeros(K)

    for i, c_i in enumerate(idx):
        centroids[c_i] = centroids[c_i] + X[i]
        Ck[c_i] += 1
    
    for i, c in enumerate(centroids):
        c /= Ck[i]
    
    return centroids

def pca(X):
    m, n = X.shape
    U = np.zeros(n)
    S = np.zeros(n)

    covariance = (X.T@X) / m
    U, S, V = np.linalg.svd(covariance)

    return U, S

def projectData(X, U, K):
    return X @ U[:, :K]

def recoverData(Z, U, K):
    return Z @ (U[:, :K]).T
