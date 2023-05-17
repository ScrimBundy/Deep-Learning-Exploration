package ActivationFunctions;

import org.apache.commons.math4.legacy.analysis.UnivariateFunction;

/**
 * An activation function that utilizes the leaky rectified linear unit.
 */
public class LeakyReLU implements ActivationFunction {

    /**
     * UnivariateFunction container for function
     */
    private Function function;

    /**
     * UnivariateFunction container for derivative
     */
    private DerivativeFunction derivative;

    /**
     * Constructor to set coefficient
     * @param coefficient term applied to input when less than 0
     */
    public LeakyReLU(double coefficient) {
        this.function = new Function(coefficient);
        this.derivative = new DerivativeFunction(coefficient);
    }

    /**
     * Gets the activation function.
     * @return activation function
     */
    public UnivariateFunction getFunction() {
        return this.function;
    }

    /**
     * Gets the derivative of the activation function.
     * @return activation function derivative
     */
    public UnivariateFunction getDerivative() {
        return this.derivative;
    }

    private class Function implements UnivariateFunction {
        /**
         * Term applied to input when the input is less than 0.
         * This term is immutable.
         */
        private double coefficient;

        public Function(double coefficient) {
            this.coefficient = coefficient;
        }

        /**
         * Applies a Leaky ReLU activation function to the input.
         * @param x input to apply the function to
         * @return Leaky ReLU applied on input
         */
        public double value(double x) {
            if(x < 0) {
                return coefficient * x;
            } else {
                return x;
            }
        }
    }

    private class DerivativeFunction implements UnivariateFunction {
        /**
         * Term applied to input when the input is less than 0.
         * This term is immutable.
         */
        private double coefficient;

        public DerivativeFunction(double coefficient) {
            this.coefficient = coefficient;
        }

        /**
         *
         * @param x input to apply the derivative to
         * @return
         */
        public double value(double x) {
            if(x < 0) {
                return coefficient;
            } else {
                return 1;
            }
        }
    }
}
