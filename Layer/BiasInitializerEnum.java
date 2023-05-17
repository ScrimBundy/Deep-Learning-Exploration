package Layer;

public enum BiasInitializerEnum {

    /**
     * Sets all weights to zero.
     */
    Zero

    /**
     * Random bias values.
     */
    ,Random

    /**
     * Uses a uniform variance so the gradient does not get too large.
     */
    ,Xavier

    /**
     * Uses Kaiming's method of weight initialization.
     */
    ,KaimingHe

    //,Uniform

    //,Orthogonal
}
