# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import numpy as np
import torch
from torch import nn
import pennylane as qml


class HybridNN(nn.Module):
    def __init__(self, input_shape, output_shape, qubits, layers, softmax_output=False, forward_state=True, ansatz='iqp', gates='rot'):
        super().__init__()
        self.input_dim = np.prod(input_shape)
        self.output_dim = np.prod(output_shape)
        self.forward_state = forward_state
        layers = [nn.Linear(self.input_dim, qubits, bias=True),
                  quantumNN(qubits, layers, ansatz=ansatz, gates=gates),
                  nn.Linear(qubits, self.output_dim, bias=True)]
        if softmax_output:
            layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*layers)

    def forward(self, obs, state=None, info=None):
        info = {} if info is None else info
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        if not self.forward_state:
            return logits
        return logits, state


def quantumNN(qubits, layers, ansatz, gates):

    dev = qml.device('default.qubit', wires=qubits)

    def qml_XYZ(a, b, c, qubit):
        qml.RX(a, wires=qubit)
        qml.RY(b, wires=qubit)
        qml.RZ(c, wires=qubit)

    def qml_CXYZ(a, b, c, control_qubit, target_qubit):
        qml.CRX(a, wires=[control_qubit, target_qubit])
        qml.CRY(b, wires=[control_qubit, target_qubit])
        qml.CRZ(c, wires=[control_qubit, target_qubit])

    def variational(inputs, scaling, weights, qubit, layer):
        if 'iqp' == ansatz:  # parameterized two-qubit rotations
            if 'rot' == gates:
                qml.CRot(scaling[layer, qubit, 0] * inputs[qubit] + weights[layer, qubit, 0],
                         scaling[layer, qubit, 1] * inputs[qubit] + weights[layer, qubit, 1],
                         scaling[layer, qubit, 2] * inputs[qubit] + weights[layer, qubit, 2],
                         wires=[qubit, (qubit + 1) % qubits])
            elif 'u3' == gates:
                qml.ctrl(qml.U3, qubit)(scaling[layer, qubit, 0] * inputs[qubit] + weights[layer, qubit, 0],
                                        scaling[layer, qubit, 1] * inputs[qubit] + weights[layer, qubit, 1],
                                        scaling[layer, qubit, 2] * inputs[qubit] + weights[layer, qubit, 2],
                                        wires=[(qubit + 1) % qubits])
            else:
                qml_CXYZ(scaling[layer, qubit, 0] * inputs[qubit] + weights[layer, qubit, 0],
                         scaling[layer, qubit, 1] * inputs[qubit] + weights[layer, qubit, 1],
                         scaling[layer, qubit, 2] * inputs[qubit] + weights[layer, qubit, 2],
                         qubit, (qubit + 1) % qubits)
        else:  # strongly entangled with parameterized single-qubit rotations
            if 'rot' == gates:
                qml.Rot(scaling[layer, qubit, 0] * inputs[qubit] + weights[layer, qubit, 0],
                        scaling[layer, qubit, 1] * inputs[qubit] + weights[layer, qubit, 1],
                        scaling[layer, qubit, 2] * inputs[qubit] + weights[layer, qubit, 2],
                        wires=qubit)
            elif 'u3' == gates:
                qml.U3(scaling[layer, qubit, 0] * inputs[qubit] + weights[layer, qubit, 0],
                       scaling[layer, qubit, 1] * inputs[qubit] + weights[layer, qubit, 1],
                       scaling[layer, qubit, 2] * inputs[qubit] + weights[layer, qubit, 2],
                       wires=qubit)
            else:
                qml_XYZ(scaling[layer, qubit, 0] * inputs[qubit] + weights[layer, qubit, 0],
                        scaling[layer, qubit, 1] * inputs[qubit] + weights[layer, qubit, 1],
                        scaling[layer, qubit, 2] * inputs[qubit] + weights[layer, qubit, 2],
                        qubit)

    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def qnode(inputs, scaling, weights):

        # test if parameters are provided in right format:
        #   (batch, qubits) for inputs
        #   (layers, qubits, 3) for scaling and weights
        assert qubits == inputs.shape[-1]
        assert layers == scaling.shape[0] and qubits == scaling.shape[1] and 3 == scaling.shape[2]
        assert layers == weights.shape[0] and qubits == weights.shape[1] and 3 == weights.shape[2]

        # need to permute batch dimension to end
        if 2 == len(inputs.shape):
            inputs = torch.permute(inputs, (1, 0))

        # initial equal superposition
        for qubit in range(qubits):
            qml.Hadamard(wires=qubit)

        for layer in range(layers - 1):

            # dynamic (variational) part
            for qubit in range(qubits):
                variational(inputs, scaling, weights, qubit=qubit, layer=layer)

            # static (entanglement / hadamard) part
            for qubit in range(qubits):
                if 'cx' == ansatz:
                    qml.CNOT(wires=[qubit, (qubit + 1) % qubits])
                elif 'cz' == ansatz:
                    qml.CZ(wires=[qubit, (qubit + 1) % qubits])
                else:
                    qml.Hadamard(wires=qubit)

        # final dynamic (variational) part
        for qubit in range(qubits):
            variational(inputs, scaling, weights, qubit=qubit, layer=layers - 1)

        # return Pauli-Z expectation value of all individual qubits
        return [qml.expval(qml.PauliZ(wires=qubit)) for qubit in range(qubits)]

    weights_shape = {'scaling': (layers, qubits, 3), 'weights': (layers, qubits, 3)}
    return qml.qnn.TorchLayer(qnode, weights_shape)

    # drawer = qml.draw(qnode)
    # print(drawer(np.ones((qubits,)), np.ones((layers, qubits, 3)), np.ones((layers, qubits, 3))))
