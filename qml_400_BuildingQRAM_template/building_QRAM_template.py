#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def qRAM(thetas):
    """Function that generates the superposition state explained above given the thetas angles.

    Args:
        - thetas (list(float)): list of angles to apply in the rotations.

    Returns:
        - (list(complex)): final state.
    """

    # QHACK #

    # Use this space to create auxiliary functions if you need it.

    def int_to_binary(value):
      """Function that converts an integer to a binary string."""

      bi = "{0:b}".format(value)
      if len(bi)<3:
        bi = "0"*(3-len(bi)) + bi
      return bi

    def ry_matrix(i):
      qml.RY(thetas[i], wires=0)

    # QHACK #

    dev = qml.device("default.qubit", wires=range(4))

    @qml.qnode(dev)
    def circuit():

        # QHACK #
        
        # Create your circuit: the first three qubits will refer to the index, the fourth to the RY rotation.
        
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)
        for i in range(2**3):
          U = qml.transforms.get_unitary_matrix(ry_matrix)(i)
          cv = int_to_binary(i)
          qml.ControlledQubitUnitary(U, control_wires=[0, 1, 2], wires=3, control_values=cv)
        
        # QHACK #

        return qml.state()

    return circuit()


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    thetas = np.array(inputs, dtype=float)

    output = qRAM(thetas)
    output = [float(i.real.round(6)) for i in output]
    print(*output, sep=",")
