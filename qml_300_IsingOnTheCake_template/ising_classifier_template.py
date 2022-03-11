import sys
import pennylane as qml
from pennylane import numpy as np
import pennylane.optimize as optimize

DATA_SIZE = 250


def square_loss(labels, predictions):
    """Computes the standard square loss between model predictions and true labels.

    Args:
        - labels (list(int)): True labels (1/-1 for the ordered/disordered phases)
        - predictions (list(int)): Model predictions (1/-1 for the ordered/disordered phases)

    Returns:
        - loss (float): the square loss
    """

    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss


def accuracy(labels, predictions):
    """Computes the accuracy of the model's predictions against the true labels.

    Args:
        - labels (list(int)): True labels (1/-1 for the ordered/disordered phases)
        - predictions (list(int)): Model predictions (1/-1 for the ordered/disordered phases)

    Returns:
        - acc (float): The accuracy.
    """

    acc = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            acc = acc + 1
    acc = acc / len(labels)

    return acc


def classify_ising_data(ising_configs, labels):
    """Learn the phases of the classical Ising model.

    Args:
        - ising_configs (np.ndarray): 250 rows of binary (0 and 1) Ising model configurations
        - labels (np.ndarray): 250 rows of labels (1 or -1)

    Returns:
        - predictions (list(int)): Your final model predictions

    Feel free to add any other functions than `cost` and `circuit` within the "# QHACK #" markers 
    that you might need.
    """

    # QHACK #

    num_wires = ising_configs.shape[1] 
    dev = qml.device("lightning.qubit", wires=num_wires) 

    # Define a variational circuit below with your needed arguments and return something meaningful

    @qml.qnode(dev, diff_method="adjoint")
    def circuit(weights, x):

        qml.BasisState(x, wires=range(num_wires))

        qml.StronglyEntanglingLayers(weights, wires=range(num_wires))

        return qml.expval(qml.PauliZ(0))

    # Define a cost function below with your needed arguments

    def variational_classifier(weights, bias, x):
        return circuit(weights, x) + bias
    
    def cost(weights, bias, X, Y):

        # QHACK #
        
        # Insert an expression for your model predictions here
        predictions = [variational_classifier(weights, bias, x) for x in X]
        
        # QHACK #
        return square_loss(Y, predictions) # DO NOT MODIFY this line

    # optimize your circuit here
    num_layers = 6
    weights_init = 0.6 * np.random.randn(num_layers, num_wires, 3, requires_grad=True)

    bias_init = np.array(0.0, requires_grad=True)

    opt = optimize.NesterovMomentumOptimizer(0.6)
    batch_size = 250
    weights = weights_init
    bias = bias_init
    X = ising_configs
    Y = labels
    for it in range(10):

        # Update the weights by one optimizer step
        batch_index = np.random.randint(0, len(X), (batch_size,))
        X_batch = X[batch_index]
        Y_batch = Y[batch_index]
        weights, bias, _, _ = opt.step(cost, weights, bias, X_batch, Y_batch)

        # Compute accuracy
        predictions = [np.sign(variational_classifier(weights, bias, x)) for x in X]
        # acc = accuracy(Y, predictions)
       
        # print(
        #     "Iter: {:5d} | Cost: {:0.7f} | Accuracy: {:0.7f} ".format(
        #         it + 1, cost(weights, bias, X, Y), acc
        #     )
        # )
    
    predictions = [np.sign(variational_classifier(weights, bias, x)).astype(int) for x in X]
    
    # QHACK #
    
    return predictions


if __name__ == "__main__":
    inputs = np.array(
        sys.stdin.read().split(","), dtype=int, requires_grad=False
    ).reshape(DATA_SIZE, -1)
    ising_configs = inputs[:, :-1]
    labels = inputs[:, -1]
    predictions = classify_ising_data(ising_configs, labels)
    print(*predictions, sep=",")
