package mikheev;

import java.io.Serializable;
import java.util.List;

public interface RadialBasedNetwork extends Serializable {

    double f(double[] inputVector);

    void initCentersKmeans(double[][] initVectors, double[][] allInputVectors, int maxEpoch, double n, double stopShift);

    void initCentersNoKmeans(double[][] initVectors);

    void initW();

    List<Double> trainWonly(double[][] inputVectors, double[] expResults, int maxEpoch, double n, double minError);

    double wDeriv(int i_ind, double[] inputVector);
}
