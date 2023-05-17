package Layer;

import java.util.Random;

/**
 * Weight initialization methods wrapped with a single generate function.
 */
public class WeightInitializer {

    /**
     * Generates a matrix of weight values based on the number of nodes in the layer. The method
     * of generation is determined with a {@link WeightInitializerEnum}.
     * @param inputs number of inputs to the layer
     * @param outputs number of outputs from the layer
     * @param e initialization method
     * @return matrix of initialized weights
     */
    public static double[][] generate(int inputs, int outputs, WeightInitializerEnum e) {
        switch(e) {
            case Random:
                return random(inputs, outputs);
            case Xavier:
                return xavier(inputs, outputs);
            case KaimingHe:
                return kaimingHe(inputs, outputs);
            default:
                return zero(inputs, outputs);
        }
    }

    /**
     * Initializes all weights to be zero.
     * @param inputs input nodes
     * @param outputs output nodes
     * @return weight matrix
     */
    private static double[][] zero(int inputs, int outputs) {
        double[][] data = new double[outputs][inputs];
        return data;
    }

    /**
     * Initializes weights at random using a standard normal distribution.
     * @param inputs input nodes
     * @param outputs output nodes
     * @return weight matrix
     */
    private static double[][] random(int inputs, int outputs) {
        return normal(inputs, outputs, 0, 1);
    }

    /**
     * Initializes weights at random using Xavier Glorot's initialization technique.
     * @param inputs input nodes
     * @param outputs output nodes
     * @return weight matrix
     */
    private static double[][] xavier(int inputs, int outputs) {
        double mean = 0;
        double stddev = Math.sqrt(2.0 / (inputs + outputs));
        return normal(inputs, outputs, mean, stddev);
    }

    /**
     * Initializes weights at random using Kaiming He's initialization technique.
     * @param inputs input nodes
     * @return weight matrix
     */
    private static double[][] kaimingHe(int inputs, int outputs) {
        double mean = 0;
        double stddev = Math.sqrt(2.0 / inputs);
        return normal(inputs, outputs, mean, stddev);
    }

    /**
     * Initializes weights at random using a general normal distribution.
     * Used as a helper function for other initialization methods.
     * @param inputs input nodes
     * @param outputs output nodes
     * @param mean mean
     * @param stddev standard deviation
     * @return weight matrix
     */
    private static double[][] normal(int inputs, int outputs, double mean, double stddev) {
        double[][] data = new double[outputs][inputs];
        Random r = new Random();
        for(int i = 0; i < outputs; i++) {
            for(int j = 0; j < inputs; j++) {
                data[i][j] = r.nextGaussian(mean, stddev);
            }
        }
        return data;
    }
}