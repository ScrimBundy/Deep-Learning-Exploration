package Layer;

import java.util.Random;

/**
 * Bias initialization methods wrapped with a single generate function.
 */
public class BiasInitializer {

    /**
     * Generates a vector of bias values based on the number of nodes in the layer. The method
     * of generation is determined with a {@link BiasInitializerEnum}.
     * @param inputs number of inputs to the layer
     * @param outputs number of outputs from the layer
     * @param e initialization method
     * @return vector of initialized biases
     */
    public static double[] generate(int inputs, int outputs, BiasInitializerEnum e) {
        switch(e) {
            case Random:
                return random(inputs);
            case Xavier:
                return xavier(inputs, outputs);
            case KaimingHe:
                return kaimingHe(inputs);
            default:
                return zero(inputs);
        }
    }

    /**
     * Initializes all biases to be zero.
     * @param inputs input nodes
     * @return bias vector
     */
    private static double[] zero(int inputs) {
        double[] data = new double[inputs];
        return data;
    }

    /**
     * Initializes biases at random using a standard normal distribution.
     * @param outputs output nodes
     * @return bias vector
     */
    private static double[] random(int outputs) {
        return normal(outputs, 0, 1);
    }

    /**
     * Initializes biases at random using Xavier Glorot's initialization technique.
     * @param inputs input nodes
     * @param outputs output nodes
     * @return bias vector
     */
    private static double[] xavier(int inputs, int outputs) {
        double mean = 0;
        double stddev = Math.sqrt(2.0 / (inputs + outputs));
        return normal(inputs, mean, stddev);
    }

    /**
     * Initializes biases at random using Kaiming He's initialization technique.
     * @param inputs input nodes
     * @return bias vector
     */
    private static double[] kaimingHe(int inputs) {
        double mean = 0;
        double stddev = Math.sqrt(2.0 / (inputs));
        return normal(inputs, mean, stddev);
    }

    /**
     * Initializes biases at random using a general normal distribution.
     * Used as a helper function for other initialization methods.
     * @param size bias vector length
     * @param mean mean
     * @param stddev standard deviation
     * @return bias vector
     */
    private static double[] normal(int size, double mean, double stddev) {
        double[] data = new double[size];
        Random r = new Random();
        for(int i = 0; i < size; i++) {
            data[i] = r.nextGaussian(mean, stddev);
        }
        return data;
    }
}
