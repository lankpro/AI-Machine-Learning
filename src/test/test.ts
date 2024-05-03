import * as ml from '../lib/index';

{
    // Feedforward neural network: solve XNOR problem (opposite of XOR)
    const inputs = new ml.Matrix([[0, 0], [0, 1], [1, 0], [1, 1]]);
    const targets = new ml.Matrix([[1], [0], [0], [1]]);

    const feedforwardNeuralNe