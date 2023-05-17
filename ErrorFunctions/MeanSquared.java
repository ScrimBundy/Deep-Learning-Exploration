package ErrorFunctions;

/**
 * An error function that uses the mean squared approach.
 */
public class MeanSquared implements ErrorFunction {

    /**
     * Calculate mean squared "forward" value.
     * @param y true value
     * @param x observed value
     * @return error
     */
    public double value(double y, double x) {
        return 0.5 * Math.pow(y - x, 2);
    }

    /**
     * Calculate mean squared derivative value.
     * @param y true value
     * @param x observed value
     * @return derivative of error
     */
    public double derivative(double y, double x) {
        return (y - x) * -1;
    }
}
