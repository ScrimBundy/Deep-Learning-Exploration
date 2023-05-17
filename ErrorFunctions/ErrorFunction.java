package ErrorFunctions;

/**
 * Function used to calculate the error of any output of a network.
 */
public interface ErrorFunction {

    /**
     * Calculated standard or "forward" error function.
     * @param y true value
     * @param x observed value
     * @return error
     */
    double value(double y, double x);

    /**
     * Calculated the derivative of the error function
     * @param y true value
     * @param x observed value
     * @return derivative of error
     */
    double derivative(double y, double x);

}
