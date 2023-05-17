import ActivationFunctions.ActivationFunction;
import ActivationFunctions.Sigmoid;
import ErrorFunctions.ErrorFunction;
import ErrorFunctions.MeanSquared;
import Layer.BiasInitializerEnum;
import Layer.FullyConnectedLayer;
import Layer.InputLayer;
import Layer.WeightInitializerEnum;
import org.apache.commons.math4.legacy.linear.*;

public class Testing {
    public static void main(String[] args) {
        ActivationFunction af = new Sigmoid();
        ErrorFunction ef = new MeanSquared();
        WeightInitializerEnum wInit = WeightInitializerEnum.Xavier;
        BiasInitializerEnum bInit = BiasInitializerEnum.Zero;
        double learningRate = 0.01;
        int inputSize = 784;
        int[] layerSizes = {200, 80, 10};

        Network n = new Network(af, ef, learningRate, inputSize, layerSizes, wInit, bInit);
    }

    public static void test1() {
        InputLayer in = new InputLayer(2);
        ActivationFunction sig = new Sigmoid();

        double[][] h1weights = {{0.15,0.25},{0.20,0.30},{0.35,0.35}};

        FullyConnectedLayer h1 = new FullyConnectedLayer(2,in,sig, h1weights);

        double[][] h2weights = {{0.40,0.50},{0.45,0.55},{0.6,0.6}};

        ErrorFunction ms = new MeanSquared();

        FullyConnectedLayer out = new FullyConnectedLayer(2, h1, sig, h2weights);



        double[][] input = {{0.05, 0.10},{0.05, 0.10}};
        RealMatrix v = new Array2DRowRealMatrix(input);
        double[][] real = {{0.01, 0.99},{0.01,0.99}};
        RealMatrix y = new Array2DRowRealMatrix(real);


        RealMatrix z1 = h1.forwardWeightedSum(v);
        RealMatrix a1 = h1.forwardActivation(z1);
        RealMatrix z2 = out.forwardWeightedSum(a1);
        RealMatrix a2 = out.forwardActivation(z2);
        RealMatrix dadc = errorDerivative(y,a2,ms);


        printMatrix(z1);
        printMatrix(a1);
        printMatrix(z2);
        printMatrix(a2);
        printMatrix(dadc);

        RealMatrix dc_da1 = out.backProp(dadc,a1,z2,0.5);
        printMatrix(dc_da1);

        RealMatrix temp = h1.backProp(dc_da1, v, z1, 0.5);
    }

    public static void printVector(RealVector v) {
        for(int i = 0; i < v.getDimension(); i++) {
            System.out.print(v.getEntry(i) + " ");
        }
        System.out.println();
    }

    public static void printMatrix(RealMatrix m) {
        for(int i = 0; i < m.getRowDimension(); i++) {
            for(int j = 0; j < m.getColumnDimension(); j++) {
                System.out.print(m.getEntry(i,j) + " ");
            }
            System.out.println();
        }
        System.out.println();
    }

    public static RealVector errorDerivative(RealVector y, RealVector z, ErrorFunction ef) {
        RealVector r = z.copy();
        for(int i = 0; i < r.getDimension(); i++) {
            r.setEntry(i, ef.derivative(y.getEntry(i), z.getEntry(i)));
        }
        return r;
    }

    public static RealMatrix errorDerivative(RealMatrix y, RealMatrix z, ErrorFunction ef) {
        RealMatrix r = z.copy();
        for(int i = 0; i < r.getRowDimension(); i++) {
            for(int j = 0; j < r.getColumnDimension(); j++) {
                r.setEntry(i,j, ef.derivative(y.getEntry(i,j), z.getEntry(i,j)));
            }
        }
        return r;
    }
}
