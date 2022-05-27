import numpy as np

def computeCost(X, y, theta):
    m = y.size 
    distance = y - (X @ theta)
    J = (distance @ distance) / (2*m)

    return J

# implemented in a slightly "un-vectorised" fashion just so that
# i can more easily visualise whats going on, in actuality
# we can do the calculations of the pde terms in one go
# with matrix mult which i will do in the one for 
# multivariate version below
def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    theta = theta.copy()
    
    J_history = []
    
    for i in range(num_iters):
        distance = (X @ theta) - y
        
        # theta0 pde term doesn't need multiplication by x
        # so we can just use the 1s column from X as a vector to sum all the distances
        # it's like (d1 * 1) + (d2 * 1) + .... + (dm * 1)
        # distance is m x 1 vector, 1s is m x 1 vector full of 1s
        theta0_term = distance@X[:,0]
        
        # for theta1 term, we need to multiple distance by i'th x
        # so we can use the "actual" x values vector and do the dot prod
        theta1_term = distance@X[:,1]
        
        sum = np.array([theta0_term, theta1_term])
        avg = sum/m
        theta = theta - (avg * alpha)

        J_history.append(computeCost(X, y, theta))
    
    return theta, J_history

def  featureNormalize(X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    # m = number of training set data points
    # n = number of features
    m, n = X.shape
    
    # for each feature
    for i in range(n):
        mean = np.mean(X_norm[:, i])
        std = np.std(X_norm[:, i])
        mu[i] = mean
        sigma[i] = std
        X_norm[:, i] = X_norm[:, i] - mean
        X_norm[:, i] = X_norm[:, i] / std
    
    return X_norm, mu, sigma

def computeCostMulti(X, y, theta):
    return computeCost(X, y, theta)

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m, n = X.shape
    theta = theta.copy()
    
    J_history = []
    
    for i in range(num_iters):
        distance = (X @ theta) - y
        pde_of_cost = distance @ X
        avg = pde_of_cost / m

        theta = theta - (avg * alpha)
        
        J_history.append(computeCostMulti(X, y, theta))
    
    return theta, J_history

def normalEqn(X, y):
    return np.linalg.pinv(X.transpose() @ X) @ X.transpose() @ y
