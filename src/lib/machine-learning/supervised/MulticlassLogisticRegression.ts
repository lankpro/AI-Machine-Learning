import LogisticRegression from "./LogisticRegression";
import Matrix from "../../math/linear-algebra/Matrix";

export default class MulticlassLogisticRegression {

    private numberOfEpochs = 1000;
    private batchSize = 0;
    private learningRate = 0.001;
    private regularizationFactor = 0;

    private logisticRegressions: LogisticRegression[];

    public constructor () {}

    public train (inputs: Matrix, targets: Matrix) {
        if (this.logisticRegressions === undefined) {
            this.logisticRegressions = [];

            for (let i = 0; i < targets.getColumnCount(); i++) {
                const logisticRegression = new LogisticRegression();
                logisticRegression.setNumberOfEpochs(this.numberOfEpochs);
                logisticRegression.setBatchSize(this.batchSize);
                logisticRegression.setLearningRate(this.learningRate);
                logisticRegression.setRegularizationFactor(this.regularizationFactor);
                this.logisticRegressions.push(logisticRegression);
            }
        }

        this.logisticRegressions.forEach((logisticRegression, i) => logisticRegression.train(inputs, targets.getColumn(i)));
    }

    public predict (inputs: Matrix) {
        return this.logisticRegressions.reduce((accumulatedPredict