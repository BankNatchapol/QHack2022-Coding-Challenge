#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml

graph = {
    0: [1],
    1: [0, 2, 3, 4],
    2: [1],
    3: [1],
    4: [1, 5, 7, 8],
    5: [4, 6],
    6: [5, 7],
    7: [4, 6],
    8: [4],
}


def n_swaps(cnot):
    """Count the minimum number of swaps needed to create the equivalent CNOT.

    Args:
        - cnot (qml.Operation): A CNOT gate that needs to be implemented on the hardware
        You can find out the wires on which an operator works by asking for the 'wires' attribute: 'cnot.wires'

    Returns:
        - (int): minimum number of swaps
    """

    # QHACK #

    # convert graph into adjacency matrix
    adj_matrix = np.zeros((len(graph), len(graph)))
    for i in range(len(graph)):
        for j in graph[i]:
            adj_matrix[i, j] = 1

    # convert adjacency matrix into distance matrix
    dist_matrix = np.zeros((len(graph), len(graph)))
    for i in range(len(graph)):
        for j in range(len(graph)):
            if adj_matrix[i, j] == 1:
                dist_matrix[i, j] = 1
            else:
                dist_matrix[i, j] = np.inf

    # Floyd-Warshall algorithm
    for k in range(len(graph)):
        for i in range(len(graph)):
            for j in range(len(graph)):
                if dist_matrix[i, k] + dist_matrix[k, j] < dist_matrix[i, j]:
                    dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]

    return int((dist_matrix[cnot.wires[0]][cnot.wires[1]]-1)*2)

    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    output = n_swaps(qml.CNOT(wires=[int(i) for i in inputs]))
    print(f"{output}")
