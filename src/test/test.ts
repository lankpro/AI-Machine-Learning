import * as ml from '../lib/index';

{
    // Feedforward neural network: solve XNOR problem (opposite of XOR)
    const inputs = new ml.Matrix([[0, 0], [0, 1], [1, 0], [1, 1]]);
    const targets = new ml.Matrix([[1], [0], [0], [1]]);

    const feedforwardNeuralNetwork = new ml.FeedforwardNeuralNetwork([2, 5, 1], 0);
    feedforwardNeuralNetwork.setNumberOfEpochs(1000);
    feedforwardNeuralNetwork.setLearningRate(1);

    feedforwardNeuralNetwork.train(inputs, targets);
    const predictions = feedforwardNeuralNetwork.predict(inputs);
    console.log(predictions.toArray());
    // [ [ 0.9943559154265011 ], [ 0.012148393118769857 ], [ 0.013640408487437417 ], [ 0.9816837627444868 ] ]
}

{
    // Linear Regression: y = 1000 + 200 * x
    const inputs = new ml.Matrix([[5], [7], [9], [11], [13]]);
    const targets = new ml.Matrix([[2000], [2400], [2800], [3200], [3600]]);

    const linearRegression = new ml.LinearRegression();
    linearRegression.setNumberOfEpochs(10000);
    linearRegression.setLearningRate(0.02);

    linearRegression.train(inputs, targets);
    const predictions = linearRegression.predict(inputs);
    console.log(predictions.toArray());
    // [ [ 1999.999991189672 ], [ 2399.9999948012005 ], [ 2799.999998412729 ], [ 3200.0000020242574 ], [ 3600.000005635786 ] ]
}

{
    // Logistic Regression: determine if second input is higher than first input
    const inputs = new ml.Matrix([[1000, 1100], [4500, 3000], [700, 1300], [1150, 700], [1300, 1200], [600, 650]]);
    const targets = new ml.Matrix([[1], [0], [1], [0], [0], [1]]);

    const logisticRegression = new ml.LogisticRegression();
    logistic