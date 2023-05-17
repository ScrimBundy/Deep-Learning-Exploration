package Layer;

public enum WeightInitializerEnum {
    /**
     * Sets all weights to zero.
     */
    Zero

    /**
     * Random weight values.
     */
    ,Random

    /**
     * Uses a uniform variance so the gradient does not get too large.
     */
    ,Xavier

    ,KaimingHe

    //,He

    //,Uniform

    //,Orthogonal
}
