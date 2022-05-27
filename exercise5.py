def linearRegCostFunction(X, y, theta_, lambda_=0.0):
    m = y.size # number of training examples

    theta = theta_.copy()

    distance = (X @ theta) - y
    grad_theta_zero = distance @ X / m
    
    J = (distance @ distance) / (2*m)
    
    theta[0] = 0
    
    grad_theta_j = theta * lambda_ / m

    J += (((theta @ theta) * lambda_) / (2 * m))
    grad = (grad_theta_zero + grad_theta_j)

    return J, grad


def learningCurve(X, y, Xval, yval, lambda_=0):
    m = y.size

    error_train = np.zeros(m)
    error_val   = np.zeros(m)
    
    for i in range(1, m+1):
        theta_t = utils.trainLinearReg(linearRegCostFunction, X[:i, :], y[:i], lambda_ = lambda_)
        
        j_t, grad_t = linearRegCostFunction(X[:i, :], y[:i], theta_t, 0)
        j_val_t, grad_val_t = linearRegCostFunction(Xval, yval, theta_t, 0)
        
        error_train[i-1] = j_t
        error_val[i-1] = j_val_t

    return error_train, error_val

def polyFeatures(X, p):
    X_poly = np.zeros((X.shape[0], p))

    for i in range(p):
        X_poly[:, i] = X[:, 0] ** (i+1)

    return X_poly

def validationCurve(X, y, Xval, yval):
    # Selected values of lambda (you should not change this)
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    # You need to return these variables correctly.
    error_train = np.zeros(len(lambda_vec))
    error_val = np.zeros(len(lambda_vec))

    for i in range(len(lambda_vec)):
        error_train_i, error_val_i = learningCurve(X, y, Xval, yval, lambda_=lambda_vec[i])
        error_train[i] = error_train_i[-1]
        error_val[i] = error_val_i[-1]

    return lambda_vec, error_train, error_val
