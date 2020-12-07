package mikheev;

import org.knowm.xchart.QuickChart;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class RBNbasic {
    public double[][] centers;
    public double[][] sigmas;
    public double[] outWeights;
    public int inputVectorSize;
    public int hiddenLayerSize;


    public RBNbasic(int inputVectorSize, int hiddenLayerSize){
        this.inputVectorSize = inputVectorSize;
        this.hiddenLayerSize = hiddenLayerSize;
        centers = new double[hiddenLayerSize][inputVectorSize];
        outWeights = new double[hiddenLayerSize + 1];
        sigmas = new double[hiddenLayerSize][inputVectorSize];
    }

    public double f(double[] inputVector){
        double[] expfuncResults = new double[hiddenLayerSize];
        for (int i = 0; i < hiddenLayerSize; i++) {
            expfuncResults[i] = expOfu(u(inputVector, i));
        }
        double sum = outWeights[0];
        for (int i = 1; i <= hiddenLayerSize; i++) {
            sum += outWeights[i] * expfuncResults[i - 1];
        }
        return sum;
    }

    public void initCenters(double[][] initVectors, double[][] allInputVectors){
        for (int i = 0; i < hiddenLayerSize; i++) {
            for (int j = 0; j < inputVectorSize; j++) {
                centers[i][j] = initVectors[i][j];
            }
        }
        //fill Q with 0;
        double totalDelta = 10;
        double n = 0.01;
        int counter = 0;
        while (totalDelta > 0.01){
            double newTotalDelta = 0;
            for (int i = 0; i < allInputVectors.length; i++){
                int winnerId = -1;
                double winnerDiff = Integer.MAX_VALUE;
                for (int j = 0; j < hiddenLayerSize; j++) {
                    double distance = 0;
                    for (int k = 0; k < inputVectorSize; k++) {
                        distance += Math.pow(allInputVectors[i][k] - centers[j][k], 2);
                    }
                    distance = Math.sqrt(distance);
                    if (distance < winnerDiff){
                        winnerDiff = distance;
                        winnerId = j;
                    }
                }
                //adjust winner
                for (int j = 0; j < inputVectorSize; j++) {
                    centers[winnerId][j] -= (n * (allInputVectors[i][j] - centers[winnerId][j]));
                    newTotalDelta += Math.abs(n * (allInputVectors[i][j] - centers[winnerId][j]));
                }
            }
            totalDelta = newTotalDelta;
            counter++;
            n /= 2;
        }

        double[][] distanceVector = new double[hiddenLayerSize][hiddenLayerSize];
        for (int i = 0; i < hiddenLayerSize; i++) {
            for (int j = 0; j < hiddenLayerSize; j++) {
                distanceVector[i][j] = 0;
                for (int k = 0; k < inputVectorSize; k++) {
                    distanceVector[i][j] += Math.pow(centers[i][k] - centers[j][k], 2);
                }
            }
            Arrays.parallelSort(distanceVector[i]);
            double radius = 0;
            //0 is at 0
            for(int j = 1; j < Math.min(hiddenLayerSize, 4); j++){
                radius += distanceVector[i][j];
            }
            radius /= (Math.min(hiddenLayerSize, 4) - 1);
            radius = Math.sqrt(radius);
            for (int k = 0; k < inputVectorSize; k++) {
                sigmas[i][k] = radius;
            }
        }
    }

    public void initW(){
        for (int i = 0; i < hiddenLayerSize + 1; i++) {
            outWeights[i] = Math.random();
        }
    }

    public double u(double[] inputVector, int i){
        double acc = 0;
        for (int j = 0; j < inputVectorSize; j++) {
            acc += Math.pow((inputVector[j] - centers[i][j]) / sigmas[i][j], 2);
        }
        return acc;
    }

    public double expOfu(double u){
        return Math.exp(-0.5 * u);
    }

    public void train(double[][] inputVectors, double[] expResults, int maxEpoch, double n, double minError){
        double oldError = Integer.MAX_VALUE;
        List<Double> inputParams = new ArrayList<>();
        List<Double> xAxis = new ArrayList<>();
        for (int i = 0; i < maxEpoch; i++) {
            double newError = 0;
            for (int j = 0; j < inputVectors.length; j++) {
                double result = f(inputVectors[j]);
                double diff = result - expResults[j];
                adjustOnlyW(diff, inputVectors[j], n);
                newError += Math.pow(diff, 2);
            }
            newError /= (inputVectors.length - 1);
            newError = Math.sqrt(newError);
            oldError = newError;
            inputParams.add(newError);
            xAxis.add((double)i);
        }
        System.out.println("lastError " + oldError);
        XYChart chart = QuickChart.getChart("error", "epoch", "errorVal", "error", xAxis, inputParams);
        new SwingWrapper<XYChart>(chart).displayChart();
    }

    public void adjustOnlyW(double diff, double[] inputVector, double n){
        double w0Deriv = diff;
        double[][] zir = new double[hiddenLayerSize][inputVectorSize];
        double[] wDeriv = new double[hiddenLayerSize];

        for (int i = 0; i < hiddenLayerSize; i++) {
            wDeriv[i] = expOfu(u(inputVector, i)) * diff;
        }

        for (int i = 0; i < hiddenLayerSize + 1; i++) {
            if (i == 0) {
                outWeights[i] = outWeights[i] - n * w0Deriv;
            } else {
                outWeights[i] = outWeights[i] - n * wDeriv[i - 1];
            }
        }
    }


    //individual derivatives for testing
    public double wDeriv(int i_ind, double[] inputVector){
        return expOfu(u(inputVector, i_ind));
    }

}

