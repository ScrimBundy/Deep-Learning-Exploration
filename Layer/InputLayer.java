package Layer;

/**
 * Layer that acts as a placeholder in place of the actual input.
 * This is primarily used to determine the dimensions of the
 * weight matrix in succeeding layers.
 */
public class InputLayer implements Layer {

    /**
     * Size of the input.
     */
    private int size;

    /**
     * Initialize an input layer.
     * @param size number of elements in input
     */
    public InputLayer(int size) {
        this.size = size;
    }

    /**
     * The number of elements in the input.
     * @return size of input
     */
    public int size() {
        return this.size;
    }
}
