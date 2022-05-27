import numpy as np

def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, lambda_=0.0):

    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

    m = y.size
         
    # add the bias column to input layer matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    # make m x n matrix of row vectors that correspond to the y labels
    # we can grab rows from an identity matrix for this
    eye = np.eye(num_labels)
    Y = np.array([eye[x] for x in y])

    # calculate activation values of hidden layer
    layer2_z = X @ Theta1.T
    layer2_activations = utils.sigmoid(layer2_z)
    # add bias
    layer2_activations = np.concatenate([np.ones((m, 1)), layer2_activations], axis=1)

    # calculate activation values of output layer
    output_z = layer2_activations @ Theta2.T
    output_activations = utils.sigmoid(output_z)
    
    left_term = np.sum(-Y * np.log(output_activations), axis=1)
    right_term = -1 * np.sum((1 - Y) * np.log(1 - output_activations), axis=1)
    
    J = np.sum(left_term + right_term) / m    
    
    # make copy of parameter matrices and set first column
    # to zero to use for regularisation calculation
    # since we don't want to add regularisation term
    # when j=0
    T1 = Theta1.copy()
    T2 = Theta2.copy()
    
    T1[:,0] = 0
    T2[:,0] = 0
    
    T1_sq_sum = np.sum(T1 * T1)
    T2_sq_sum = np.sum(T2 * T2)

    # add regularisation term to cost
    J += ((lambda_ * (T1_sq_sum + T2_sq_sum)) / (2 * m))
    
    # Backprop
    # output layer errors
    d3 = (output_activations - Y)

    # multiply d3 with Theta2 except for the bias column
    weightedD3 = (d3 @ Theta2[:,1:])

    z2Grad = (layer2_activations * (1 - layer2_activations))

    # hidden layer errors
    d2 = (weightedD3 * z2Grad[:,1:])

    Theta1_grad = (d2.T @ X) / m
    Theta2_grad = ((d3.T @ layer2_activations)) / m
    
    # add regularisation term
    Theta1_grad += ((lambda_ * T1) / m)
    Theta2_grad += ((lambda_ * T2) / m)

    # Unroll param gradients
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])
    
    return J, grad
