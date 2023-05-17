package Layer;
import ActivationFunctions.ActivationFunction;
import org.apache.commons.math4.legacy.analysis.UnivariateFunction;
import org.apache.commons.math4.legacy.linear.*;

import java.util.Arrays;

/**
 * A network layer that is fully connected with the previous layer.
 * Fully connected implies that there exists an edge between all pairs of nodes from the previous layer to this one.
 */
public class FullyConnectedLayer implements Layer {

    /**
     * Reference to the previous node in the network.
     * Primarily used to determine the size of the weight matrix.
     */
    private Layer previousLayer;

    /**
     * Number of nodes in the layer.
     */
    private int size;

    /**
     * Weights and bias matrix.
     * Weights per node are the column vectors excluding the last row.
     * Bias per node is the last row in the matrix.
     */
    private RealMatrix weights;

    /**
     * Activation function and its derivative.
     */
    private ActivationFunction act;

    /**
     * Initializes layer with default weights and biases.
     * @param size number of "neurons"
     * @param previousLayer reference to previous layer in network
     * @param act activation function
     */
    public FullyConnectedLayer(int size, Layer previousLayer, ActivationFunction act) {
        this.size = size;
        this.previousLayer = previousLayer;
        this.act = act;
    }

    /**
     * Initializes layer with predefined weights and biases.
     * Precondition: {@code weights} must have {@code size + 1} rows and {@code previousLayer.size()} columns.
     * @param size number of "neurons"
     * @param previousLayer reference to previous layer in network
     * @param act activation function
     * @param weights predefined weights and biases
     */
    public FullyConnectedLayer(int size, Layer previousLayer, ActivationFunction act, double[][] weights) {
        this(size, previousLayer, act);
        this.weights = new Array2DRowRealMatrix(weights);
    }

    /**
     * Initializes layer utilizing the provided weight and bias initializer methods.
     * @param size number of "neurons"
     * @param previousLayer reference to previous layer in network
     * @param act activation function
     * @param wInit weight initializer
     * @param bInit bias initializer
     */
    public FullyConnectedLayer(int size, Layer previousLayer, ActivationFunction act,
                               WeightInitializerEnum wInit, BiasInitializerEnum bInit) {
        this(size, previousLayer, act);
        double[][] w = WeightInitializer.generate(previousLayer.size(), size, wInit);
        double[] b = BiasInitializer.generate(previousLayer.size(), size, bInit);

        this.weights = MatrixUtils.createRealMatrix(size + 1, previousLayer.size());
        this.weights.setSubMatrix(w, 0, 0);
        this.weights.setRow(size, b);
    }

    /**
     * Calculated the weighted sum of previous layer activation with this
     * layer's weights and biases.
     * @param input previous layer activation vector
     * @return weighted sums
     */
    public RealVector forwardWeightedSum(RealVector input) {
        return this.weights.preMultiply(input.append(1));
    }

    /**
     * Uses matrix multiplication to calculate the weighted sum of
     * multiple inputs simultaneously. The input should be structured
     * such that each row is an input vector.
     * @param input matrix of input vectors
     * @return output vectors as rows of a matrix containing the weighted sums
     */
    public RealMatrix forwardWeightedSum(RealMatrix input) {
        RealMatrix inputMod = appendColumnOfOnes(input);
        return this.weights.preMultiply(inputMod);
    }

    /**
     * Performs the activation function on each entry in the vector.
     * @param z weighted sum vector
     * @return activation values vector
     */
    public RealVector forwardActivation(RealVector z) {
        return z.map(this.act.getFunction());
    }

    /**
     * Performs the activation function on each entry in the matrix.
     * @param z weighted sum matrix
     * @return activation values matrix
     */
    public RealMatrix forwardActivation(RealMatrix z) {
        return map(z, this.act.getFunction());
    }


    /**
     * Updates the weights and biases based on the derivative of the cost/loss
     * function with respect to each node's activation value for a single
     * test case.
     * @param dc_da vector with derivative of cost/loss with respect to activation
     * @param a0 vector with previous layer activation values
     * @param z vector with weighted sums
     * @param alpha learning rate
     * @return derivative of cost/loss with respect to previous layer activation
     */
    public RealVector backProp(RealVector dc_da, RealVector a0, RealVector z, double alpha) {

        // derivative of current layer activation with respect to weighted sum
        RealVector da_dz = weightedSumDerivative(z);
        // derivative of cost with respect to weighted sum
        RealVector dc_dz = dc_da.ebeMultiply(da_dz);

        /*
        //--test prints--
        System.out.print("da_dz: ");
        printVector(da_dz);
        System.out.print("dc_dz: ");
        printVector(dc_dz);

         */

        // derivative of cost with respect to both weights and bias
        RealMatrix dc_dw = MatrixUtils.createRealMatrix(this.weights.getRowDimension(),
                                                        this.weights.getColumnDimension());
        // set weights
        dc_dw.setSubMatrix(dc_dz.outerProduct(a0).transpose().getData(), 0, 0); //check this
        // set bias
        dc_dw.setRowVector(dc_dw.getRowDimension() - 1, dc_dz);

        /*
        System.out.println("dc_dw: ");
        printMatrix(dc_dw);
        System.out.println();

         */

        // multiply by learning rate
        dc_dw = dc_dw.scalarMultiply(alpha);
        // get just weights, no bias
        RealMatrix w = this.weights.getSubMatrix(0, this.weights.getRowDimension() - 2,
                                                 0, this.weights.getColumnDimension() - 1);
        // derivative of cost with respect to previous layer activation values
        RealVector dc_da0 = w.transpose().preMultiply(dc_dz); // check this

        // adjust weights and biases
        this.weights = this.weights.subtract(dc_dw);

        /*
        System.out.println("new weights: ");
        printMatrix(this.weights);
        System.out.println();

         */

        return dc_da0;
    }

    /**
     * Debug tool. Prints a RealVector in a readable format to standard output.
     * @param v
     */
    static void printVector(RealVector v) {
        for(int i = 0; i < v.getDimension(); i++) {
            System.out.print(v.getEntry(i) + " ");
        }
        System.out.println();
    }

    /**
     * Debug tool. Prints a RealMatrix in a readable format to standard output.
     * @param m matrix
     */
    static void printMatrix(RealMatrix m) {
        for(int i = 0; i < m.getRowDimension(); i++) {
            for(int j = 0; j < m.getColumnDimension(); j++) {
                System.out.print(m.getEntry(i,j) + " ");
            }
            System.out.println();
        }
    }


    /**
     * Updates the weights and biases based on the derivative of the cost/loss
     * function with respect to each node's activation value for a set of
     * test cases.
     * @param dc_da matrix with derivative of cost/loss with respect to activation
     * @param a0 matrix with previous layer activation values
     * @param z matrix with weighted sums
     * @param alpha learning rate
     * @return derivative of cost/loss with respect to previous layer activation
     */
    public RealMatrix backProp(RealMatrix dc_da, RealMatrix a0, RealMatrix z, double alpha) {

        // derivative of current layer activation with respect to weighted sum
        RealMatrix da_dz = weightedSumDerivative(z);
        // derivative of cost with respect to weighted sum
        RealMatrix dc_dz = ebeMultiply(dc_da, da_dz);

        // AVERAGE derivative of cost with respect to both weights and bias
        RealMatrix dc_dw = MatrixUtils.createRealMatrix(this.weights.getRowDimension(),
                                                        this.weights.getColumnDimension());
        // set weights. Scalar multiply to get averages
        dc_dw.setSubMatrix(
                dc_dz.multiply(a0.transpose()).scalarMultiply(1.0 / dc_dz.getRowDimension()).getData(),
                0, 0);

        // set bias
        RealVector dc_db = new ArrayRealVector(dc_dz.getColumnDimension());
        for(int j = 0; j < dc_dz.getColumnDimension(); j++) {
            // sum over the column
            double total = 0;
            for(int i = 0; i < dc_dz.getRowDimension(); i++) {
                total += dc_dz.getEntry(i,j);
            }
            double average = total / dc_dz.getRowDimension();
            dc_db.setEntry(j, average);
        }
        dc_dw.setRowVector(dc_dw.getRowDimension() - 1, dc_db);

        // multiply by learning rate
        dc_dw = dc_dw.scalarMultiply(alpha);
        // get just weights, no bias
        RealMatrix w = this.weights.getSubMatrix(0, this.weights.getRowDimension() - 2,
                0, this.weights.getColumnDimension() - 1);
        // derivative of cost with respect to previous layer activation values
        RealMatrix dc_da0 = w.transpose().preMultiply(dc_dz); // check this

        // adjust weights and biases
        this.weights = this.weights.subtract(dc_dw);

        return dc_da0;
    }

    /**
     * Calculates the derivative of activation with respect to the weighted sum.
     * @param z weighted sum vector
     * @return derivative of activation with respect to weighted sum
     */
    private RealVector weightedSumDerivative(RealVector z) {
        return z.map(this.act.getDerivative());
    }

    /**
     * Calculates the derivative of activation with respect to the weighted sum
     * for multiple test cases.
     * @param z wighted sum matrix
     * @return derivative of activation with respect to weighted sum
     */
    private RealMatrix weightedSumDerivative(RealMatrix z) {
        return map(z, this.act.getDerivative());
    }

    /**
     * Maps the given matrix to a new matrix with the given function.
     *
     * Returns a new matrix and does not modify instance data.
     * @param m matrix to map.
     * @param f Function to apply to each entry.
     * @return a new matrix.
     */
    private RealMatrix map(RealMatrix m, UnivariateFunction f) {
        int rows = m.getRowDimension();
        int cols = m.getColumnDimension();
        RealMatrix r = MatrixUtils.createRealMatrix(rows,cols);

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                r.setEntry(i, j, f.value(m.getEntry(i,j)));
            }
        }
        return r;
    }

    /**
     * Element-by-element multiplies two matrices.
     *
     * Precondition: both matrices must have the same dimensions.
     * @param a first matrix
     * @param b second matrix
     * @return a new matrix
     */
    private RealMatrix ebeMultiply(RealMatrix a, RealMatrix b) {
        int rows = a.getRowDimension();
        int cols = a.getColumnDimension();
        RealMatrix r = MatrixUtils.createRealMatrix(rows, cols);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                r.setEntry(i, j, a.getEntry(i,j) * b.getEntry(i,j));
            }
        }
        return r;
    }

    /**
     * Helper function to append a column consisting of ones to the end
     * of a matrix.
     * @param m any matrix
     * @return copy of m with a column of ones appended to the end
     */
    public RealMatrix appendColumnOfOnes(RealMatrix m) {
        // Create new matrix with an additional column
        RealMatrix m2 = MatrixUtils.createRealMatrix(m.getRowDimension(), m.getColumnDimension() + 1);
        // Copy data to new matrix
        m2.setSubMatrix(m.getData(),0,0);
        // Create array of ones
        double[] onesColumn = new double[m2.getRowDimension()];
        Arrays.fill(onesColumn,1.0);
        // Set the new column to ones
        m2.setColumn(m2.getColumnDimension() - 1, onesColumn);

        return m2;
    }

    /**
     * Gets the number of nodes in this layer.
     * @return size
     */
    public int size() {
        return this.size;
    }

}