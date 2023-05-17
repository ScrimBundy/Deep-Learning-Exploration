import ActivationFunctions.ActivationFunction;
import ErrorFunctions.ErrorFunction;
import Layer.FullyConnectedLayer;
import Layer.InputLayer;
import Layer.Layer;
import Layer.WeightInitializerEnum;
import Layer.BiasInitializerEnum;
import org.apache.commons.math4.legacy.linear.RealMatrix;
import org.apache.commons.math4.legacy.linear.RealVector;

import java.io.Serializable;

public class Network implements Serializable {

    private InputLayer inL;
    private FullyConnectedLayer[] hL;
    private ActivationFunction af;
    private ErrorFunction ef;
    private double learnRate;

    /**
     *
     * @param af
     * @param ef
     * @param layerSizes array containing two or more layer sizes
     */
    public Network(ActivationFunction af, ErrorFunction ef, double learnRate, int inputSize, int[] layerSizes) {
        this.af = af;
        this.ef = ef;
        this.learnRate = learnRate;

        this.inL = new InputLayer(inputSize);
        this.hL = new FullyConnectedLayer[layerSizes.length];

        Layer prev = this.inL;

        for(int i = 0; i < layerSizes.length; i++) {
            this.hL[i] = new FullyConnectedLayer(layerSizes[i], prev, af);
            prev = this.hL[i];
        }
    }

    public Network(ActivationFunction af, ErrorFunction ef, double learnRate, int inputSize, int[] layerSizes,
                   WeightInitializerEnum wInit, BiasInitializerEnum bInit) {
        this.af = af;
        this.ef = ef;
        this.learnRate = learnRate;

        this.inL = new InputLayer(inputSize);
        this.hL = new FullyConnectedLayer[layerSizes.length];

        Layer prev = this.inL;

        for(int i = 0; i < layerSizes.length; i++) {
            this.hL[i] = new FullyConnectedLayer(layerSizes[i], prev, af, wInit, bInit);
            prev = this.hL[i];
        }
    }

    /**
     * Feed an input into the network.
     * This method retains none of the information needed to perform
     * back-propagation on this input data.
     * @param input row vector of input. Length must be equal to inputSize.
     * @return activation values of the final layer
     */
    public RealVector forwardPass(RealVector input) {
        RealVector prevAct = input.copy();
        for(int i = 0; i < this.hL.length; i++) {
            RealVector z = hL[i].forwardWeightedSum(prevAct);
            RealVector a = hL[i].forwardActivation(z);
            prevAct = a;
        }
        return prevAct;
    }

    /**
     * Feed several inputs into the network.
     * This method retains none of the information needed to perform
     * back-propagation on this input data.
     * @param input matrix of input row vectors. Must have inputSize columns.
     * @return activation values of the final layer per input
     */
    public RealMatrix forwardPass(RealMatrix input) {
        RealMatrix prevAct = input.copy();
        for(int i = 0; i < this.hL.length; i++) {
            RealMatrix z = hL[i].forwardWeightedSum(prevAct);
            RealMatrix a = hL[i].forwardActivation(z);
            prevAct = a;
        }
        return prevAct;
    }

    /**
     * Given a single test input, perform stochastic gradient descent.
     * Updates all weights and biases of all hidden layers.
     * @param input row vector of input. Length must be equal to inputSize.
     * @param expected row vector of expected values. Length must be equal to inputSize.
     * @return total error of the input
     */
    public double backPropagation(RealVector input, RealVector expected) {
        int sz = hL.length + 1;
        RealVector[] activations = new RealVector[sz];
        RealVector[] weightedSums = new RealVector[sz];

        activations[0] = input.copy();

        // forward pass
        for(int i = 0; i < this.hL.length; i++) {
            RealVector z = hL[i].forwardWeightedSum(activations[i]);
            RealVector a = hL[i].forwardActivation(z);
            weightedSums[i + 1] = z;
            activations[i + 1] = a;
        }

        // calculate error
        RealVector dc_da = errorDerivative(expected, activations[sz - 1]);
        double totalError = 0;
        for(int i = 0; i < dc_da.getDimension(); i++) {
            totalError += dc_da.getEntry(i);
        }
        totalError /= dc_da.getDimension();

        // backwards pass
        for(int i = this.hL.length - 1; i >= 0; i--) {
            dc_da = hL[i].backProp(dc_da, activations[i], weightedSums[i + 1], this.learnRate);
        }


        return totalError;
    }

    /**
     * Given a multiple test inputs, perform stochastic gradient descent.
     * Updates all weights and biases of all hidden layers.
     * @param input matrix of input row vectors. Must have inputSize columns.
     * @param expected matrix of expected value row vectors. Must have inputSize columns.
     * @return average error of the inputs
     */
    public double backPropagation(RealMatrix input, RealMatrix expected) {
        int sz = hL.length + 1;
        RealMatrix[] activations = new RealMatrix[sz];
        RealMatrix[] weightedSums = new RealMatrix[sz];

        activations[0] = input.copy();

        // forward pass
        for(int i = 0; i < this.hL.length; i++) {
            RealMatrix z = hL[i].forwardWeightedSum(activations[i]);
            RealMatrix a = hL[i].forwardActivation(z);
            weightedSums[i + 1] = z;
            activations[i + 1] = a;
        }

        // calculate error
        RealMatrix dc_da = errorDerivative(expected, activations[sz - 1]);
        double totalError = 0;
        for(int i = 0; i < dc_da.getRowDimension(); i++) {
            for(int j = 0; j < dc_da.getColumnDimension(); j++) {
                totalError += dc_da.getEntry(i,j);
            }
        }
        totalError /= (dc_da.getRowDimension() * dc_da.getColumnDimension());

        // backwards pass
        for(int i = this.hL.length - 1; i >= 0; i--) {
            dc_da = hL[i].backProp(dc_da, activations[i], weightedSums[i + 1], this.learnRate);
        }

        return totalError;
    }

    /**
     * Use error function to determine error vector.
     * @param y expected values
     * @param z actual values
     * @return error
     */
    private RealVector errorDerivative(RealVector y, RealVector z) {
        RealVector r = z.copy();
        for(int i = 0; i < r.getDimension(); i++) {
            r.setEntry(i, ef.derivative(y.getEntry(i), z.getEntry(i)));
        }
        return r;
    }

    /**
     * Use error function to determine error matrix.
     * @param y expected values
     * @param z actual values
     * @return error
     */
    private RealMatrix errorDerivative(RealMatrix y, RealMatrix z) {
        RealMatrix r = z.copy();
        for(int i = 0; i < r.getRowDimension(); i++) {
            for(int j = 0; j < r.getColumnDimension(); j++) {
                r.setEntry(i,j, ef.derivative(y.getEntry(i,j), z.getEntry(i,j)));
            }
        }
        return r;
    }
}
