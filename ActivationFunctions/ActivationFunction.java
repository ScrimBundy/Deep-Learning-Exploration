package ActivationFunctions;

import org.apache.commons.math4.legacy.analysis.UnivariateFunction;

/**
 * Represents an activation function.
 * This function should be non-linear.
 */
public interface ActivationFunction {
    /**
     * Function.
     */
    UnivariateFunction getFunction();

    /**
     * Derivative of the function.
     */
    UnivariateFunction getDerivative();
}
