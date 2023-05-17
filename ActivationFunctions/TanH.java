package ActivationFunctions;

import org.apache.commons.math4.legacy.analysis.UnivariateFunction;

/**
 * An activation function that utilizes the hyperbolic tangent function.
 */
public class TanH implements ActivationFunction {
    /**
     * UnivariateFunction container for function
     */
    private Function function;

    /**
     * UnivariateFunction container for derivative
     */
    private DerivativeFunction derivative;

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
         * Applies a hyperbolic tangent activation function to the input.
         * @param x input to apply the function to
         * @return tanH applied on input
         */
        public double value(double x) {
            return Math.tanh(x);
        }
    }

    private class DerivativeFunction implements UnivariateFunction {
        /**
         *
         * @param x input to apply the derivative to
         * @return
         */
        public double value(double x) {
            double fx = tanH(x);
            return 1 - (fx * fx);
        }

        /**
         * Copy of the function for use in its own derivative.
         * @param x value
         * @return tanH applied to value
         */
        private double tanH(double x) {
            return Math.tanh(x);
        }
    }
}
