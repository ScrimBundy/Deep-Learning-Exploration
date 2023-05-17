package ActivationFunctions;

import org.apache.commons.math4.legacy.analysis.UnivariateFunction;

/**
 * An activation function that utilizes the sigmoid function.
 */
public class Sigmoid implements ActivationFunction {
    /**
     * UnivariateFunction container for function
     */
    private Function function;

    /**
     * UnivariateFunction container for derivative
     */
    private DerivativeFunction derivative;

    public Sigmoid() {
        this.function = new Function();
        this.derivative = new DerivativeFunction();
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
         * Applies a sigmoid activation function to the input.
         * @param x input to apply the function to
         * @return sigmoid applied on input
         */
        public double value(double x) {
            return 1 / (1 + Math.exp(-x));
        }
    }

    private class DerivativeFunction implements UnivariateFunction {
        /**
         *
         * @param x input to apply the derivative to
         * @return
         */
        public double value(double x) {
            double fx = sigmoid(x);
            return fx * (1 - fx);
        }

        /**
         * Copy of the function for use in its own derivative.
         * @param x value
         * @return sigmoid applied to value
         */
        private double sigmoid(double x) {
            return 1 / (1 + Math.exp(-x));
        }
    }
}
