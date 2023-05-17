package ActivationFunctions;

import org.apache.commons.math4.legacy.analysis.UnivariateFunction;

/**
 * An activation function that utilizes the rectified linear unit.
 */
public class ReLU implements ActivationFunction {

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
         * Applies a ReLU activation function to the input.
         *
         * @param x input to apply the function to
         * @return ReLU applied on input
         */
        public double value(double x) {
            return Math.max(0, x);
        }
    }

    private class DerivativeFunction implements UnivariateFunction {
        /**
         *
         * @param x input to apply the derivative to
         * @return
         */
        public double value(double x) {
            if(x < 0) {
                return 0;
            } else {
                return 1;
            }
        }
    }
}
