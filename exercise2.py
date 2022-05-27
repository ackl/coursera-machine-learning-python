import numpy as np

def sigmoid(z):
    # convert input to a numpy array
    z = np.array(z)
    exp_term = np.exp(-z)
    return 1 / (1 + exp_term)

def costFunction(theta, X, y):
    m = y.size  # number of training examples

    h_theta = sigmoid(X @ theta)

    left_term = y * np.log(h_theta)
    right_term = (1 - y) * np.log(1 - h_theta) 

    J = np.sum(-left_term - right_term) / m
    
    grad = ((h_theta - y) @ X) / m

    return J, grad

def predict(theta, X):
    return np.where(sigmoid(X @ theta) >= 0.5, 1, 0)

def costFunctionReg(theta_, X, y, lambda_):
    m = y.size  # number of training examples
    theta = theta_.copy()

    h_theta = sigmoid(X @ theta_)
    left_term = y * np.log(h_theta)
    right_term = (1 - y) * np.log(1 - h_theta) 
    J = np.sum(-left_term - right_term) / m
    
    # since we don't want to regularize theta_0
    # just set our copy of it to zero so that the
    # lambda / m term cancels out
    theta[0] = 0
    
    J += (((theta @ theta) * lambda_) / (2 * m))
    
    grad = ((h_theta - y) @ X) / m
    grad += (lambda_ * theta) / m
    
    return J, grad
