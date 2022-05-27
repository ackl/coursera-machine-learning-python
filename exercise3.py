import numpy as np
from scipy import optimize

# same as the regularized cost function from last exercise...
def lrCostFunction(theta_, X, y, lambda_):
    m = y.size
    
    # convert labels to ints if their type is bool
    if y.dtype == bool:
        y = y.astype(int)
    

    theta = theta_.copy()

    h_theta = utils.sigmoid(X @ theta_)
    left_term = y * np.log(h_theta)
    right_term = (1 - y) * np.log(1 - h_theta) 
    J = np.sum(-left_term - right_term) / m
    
    theta[0] = 0
    
    J += (((theta@theta) * lambda_) / (2 * m))
    
    grad = ((h_theta - y) @ X) / m
    grad += (lambda_ * theta) / m
        
    return J, grad

def oneVsAll(X, y, num_labels, lambda_):
    m, n = X.shape
    
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    for i in range(num_labels):
        initial_theta = np.zeros(n + 1)
        options = {'maxiter': 50}
        res = optimize.minimize(lrCostFunction, 
                                initial_theta, 
                                (X, (y == i), lambda_), 
                                jac=True, 
                                method='TNC',
                                options=options) 
        all_theta[i] = res.x

    return all_theta

def predictOneVsAll(all_theta, X):
    m = X.shape[0];
    num_labels = all_theta.shape[0]

    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    p = np.argmax(X @ all_theta.T, axis=1)
    # we mat mult the thetas vector with transpose of X
    # then take the transpose of that result
    # which is the vectorised equivalient of below:

    #     for i in range(m):
    #         p[i] = np.argmax(all_theta @ X[i])

    return p

def predict(Theta1, Theta2, X):
    # Make sure the input has two dimensions
    if X.ndim == 1:
        X = X[None]  # promote to 2-dimensions

    m = X.shape[0]

    # add bias to input layer
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    layer2_z = (Theta1 @ X.T).T
    layer2_activations = utils.sigmoid(layer2_z)

    # add bias to layer 2
    layer2_activations = np.concatenate([np.ones((m, 1)), layer2_activations], axis=1)
    
    output_z = (Theta2 @ layer2_activations.T).T
    output_activations = utils.sigmoid(output_z)
    return np.argmax(output_activations, axis=1)
