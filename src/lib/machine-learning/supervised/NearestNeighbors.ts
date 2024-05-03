import Matrix from "../../math/linear-algebra/Matrix";

export default class NearestNeighbors {

    private distanceFunction = (x: Matrix, y: Matrix) => Matrix.subtract(x, y).transform(value => value * value).getSum();
    private numberOfNeighbors = 1;

    private inputs: Matrix;
    private targets: Matrix;

    public constructor () {}

    public train (inputs: Matrix, targets: Matrix) {
        this.inputs = inputs;
        this.targets = targets;
    }

    public predict (inputs: Matrix) {
        const outputs = new Matrix([]);

        for (let i = 0; i < inputs.getRowCount(); i++) {
            outputs.appendBottom(this.predictOne(inputs.getRow(i)));
        }

        return outputs;
    }

    /* Parameter setters */

    public setDistanceFunction (distanceFunction: (x: Matrix, y: Matrix) => number) {
        this.distanceFunction = distanceFunction;
    }

    public setNumberOfNeighbors (numberOfNeighbors: number) {
        this.numberOfNeighbors = numberOfNeighbors;
    }

    /* Parameter getters */

    public getDistanceFunction () {
        return this.distanceFunction;
    }

    public getNumberOfNeighbors () {
        return this.numberOfNeighbors;
    }

    /* Private methods */

   