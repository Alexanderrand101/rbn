package mikheev;

import org.knowm.xchart.QuickChart;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class RBN {
    public double[][] centers;
    public double[][][] correlationMatrices;
    public double[] outWeights;
    public int inputVectorSize;
    public int hiddenLayerSize;

    public static double[][] matrixMul(double[][] a, double[][] b){
        double[][] resMat = new double[a.length][b[0].length];
        for (int i = 0; i < resMat.length; i++) {
            for (int j = 0; j < resMat[0].length; j++) {
                resMat[i][j] = 0;
                for (int k = 0; k < a[0].length; k++) {
                    resMat[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return resMat;
    }

    public static double[][] matrixTranspose(double[][] a){
        double[][] resMat = new double[a[0].length][a.length];
        for (int i = 0; i < resMat.length; i++) {
            for (int j = 0; j < resMat[0].length; j++) {
                resMat[i][j] = a[j][i];
            }
        }
        return resMat;
    }

    public RBN(int inputVectorSize, int hiddenLayerSize){
        this.inputVectorSize = inputVectorSize;
        this.hiddenLayerSize = hiddenLayerSize;
        centers = new double[hiddenLayerSize][inputVectorSize];
        outWeights = new double[hiddenLayerSize + 1];
        correlationMatrices = new double[hiddenLayerSize][inputVectorSize][inputVectorSize];
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

    public void initCenters(double[][] initVectors){
        for (int i = 0; i < hiddenLayerSize; i++) {
            for (int j = 0; j < inputVectorSize; j++) {
                centers[i][j] = initVectors[i][j];
            }
        }
        //fill Q with 0;
        for (int i = 0; i < hiddenLayerSize; i++) {
            for (int r = 0; r < inputVectorSize; r++) {
                for (int j = 0; j < inputVectorSize; j++) {
                    correlationMatrices[i][r][j] = 0;//Math.random();
                }
            }
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
            radius = Math.sqrt(2 * radius);
            for (int k = 0; k < inputVectorSize; k++) {
                correlationMatrices[i][k][k] = 1/radius;
            }
        }
    }

    public void initW(){
        for (int i = 0; i < hiddenLayerSize + 1; i++) {
            outWeights[i] = Math.random();
        }
    }

    public double u(double[] inputVector, int i){
        double[][] xminusc = new double[inputVectorSize][1];
        for (int j = 0; j < inputVectorSize; j++) {
            xminusc[j][0] = inputVector[j] - centers[i][j];
        }
        double[][] qbyxminusc = matrixMul(correlationMatrices[i], xminusc);
        double[][] qbyxminuscTransposed = matrixTranspose(qbyxminusc);
        double[][] resval = matrixMul(qbyxminuscTransposed, qbyxminusc);
        return resval[0][0];
        /*double u = 0;
        for (int r = 0; r < inputVectorSize; r++) {
            double z = 0;
            for (int j = 0; j < inputVectorSize; j++) {
                z += correlationMatrices[i][r][j] * (inputVector[j] - centers[i][j]);
            }
            u += (z * z);
        }
        return u;*/
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
                double vectError = 0;
                double diff = result - expResults[j];
                vectError += Math.pow(diff, 2);
                newError += vectError;
                adjustCoeffs(diff, inputVectors[j], n, i);
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

    public void train2(double[][] inputVectors, double[] expResults, int maxEpoch, double n, double minError){
        double oldError = Integer.MAX_VALUE;
        List<Double> inputParams = new ArrayList<>();
        List<Double> xAxis = new ArrayList<>();
        for (int i = 0; i < maxEpoch; i++) {
            double newError = 0;
            for (int j = 0; j < inputVectors.length; j++) {
                double result = f(inputVectors[j]);
                double vectError = 0;
                double diff = result - expResults[j];
                adjustOnlyW(diff, inputVectors[j], n);
                result = f(inputVectors[j]);
                diff = result - expResults[j];
                adjustOnlyCandQ(diff, inputVectors[j], n);
                result = f(inputVectors[j]);
                diff = result - expResults[j];
                vectError += Math.pow(diff, 2);
                newError += vectError;
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

    public void adjustCoeffs(double diff, double[] inputVector, double n, int epoch){
        double w0Deriv = diff;
        double[] wDeriv = new double[hiddenLayerSize];
        double[][] cDeriv = new double[hiddenLayerSize][inputVectorSize];
        double[][][] Qderiv = new double[hiddenLayerSize][inputVectorSize][inputVectorSize];
        double[][] zir = new double[hiddenLayerSize][inputVectorSize];
        double[] ui = new double[hiddenLayerSize];

        for (int i = 0; i < hiddenLayerSize; i++) {
            ui[i] = 0;
            for (int r = 0; r < inputVectorSize; r++) {
                zir[i][r] = 0;
                for (int j = 0; j < inputVectorSize; j++) {
                    zir[i][r] += correlationMatrices[i][r][j] * (inputVector[j] - centers[i][j]);
                }
                ui[i] += (zir[i][r] * zir[i][r]);
            }
        }

        for (int i = 0; i < hiddenLayerSize; i++) {
            wDeriv[i] = expOfu(ui[i]) * diff;
        }

        for (int i = 0; i < hiddenLayerSize; i++) {
            double diffWSum = outWeights[i + 1] * diff;
            for (int j = 0; j < inputVectorSize; j++) {
                double zQsum = 0;
                for (int r = 0; r < inputVectorSize; r++) {
                    zQsum += correlationMatrices[i][r][j] * zir[i][r];
                }
                cDeriv[i][j] = 1 * expOfu(ui[i]) * diffWSum * zQsum;
            }
        }

        for (int i = 0; i < hiddenLayerSize; i++) {
            double diffWSum = outWeights[i + 1] * diff;
            for (int r = 0; r < inputVectorSize; r++) {
                for (int j = 0; j < inputVectorSize; j++) {
                    Qderiv[i][r][j] = -1 * expOfu(ui[i]) * diffWSum * (inputVector[j] - centers[i][j]) * zir[i][r];
                }
            }
        }

        for (int i = 0; i < hiddenLayerSize + 1; i++) {
            if (i == 0) {
                outWeights[i] = outWeights[i] - n * w0Deriv;
            } else {
                outWeights[i] = outWeights[i] - n * wDeriv[i - 1];
            }
        }

        for (int i = 0; i < hiddenLayerSize; i++) {
            for (int j = 0; j < inputVectorSize; j++) {
                centers[i][j] = centers[i][j] - n * cDeriv[i][j];
            }
        }

        for (int i = 0; i < hiddenLayerSize; i++) {
            for (int r = 0; r < inputVectorSize; r++) {
                //for (int j = 0; j < inputVectorSize; j++) {
                    correlationMatrices[i][r][r] = correlationMatrices[i][r][r] - n * Qderiv[i][r][r];
                //}
            }
        }
    }

    public void adjustOnlyW(double diff, double[] inputVector, double n){
        double w0Deriv = diff;
        double[][] zir = new double[hiddenLayerSize][inputVectorSize];
        double[] ui = new double[hiddenLayerSize];
        double[] wDeriv = new double[hiddenLayerSize];
        for (int i = 0; i < hiddenLayerSize; i++) {
            ui[i] = 0;
            for (int r = 0; r < inputVectorSize; r++) {
                zir[i][r] = 0;
                for (int j = 0; j < inputVectorSize; j++) {
                    zir[i][r] += correlationMatrices[i][r][j] * (inputVector[j] - centers[i][j]);
                }
                ui[i] += (zir[i][r] * zir[i][r]);
            }
        }

        for (int i = 0; i < hiddenLayerSize; i++) {
            wDeriv[i] = expOfu(ui[i]) * diff;
        }

        for (int i = 0; i < hiddenLayerSize + 1; i++) {
            if (i == 0) {
                outWeights[i] = outWeights[i] - n * w0Deriv;
            } else {
                outWeights[i] = outWeights[i] - n * wDeriv[i - 1];
            }
        }
    }

    public void adjustOnlyCandQ(double diff, double[] inputVector, double n){
        double[][] cDeriv = new double[hiddenLayerSize][inputVectorSize];
        double[][][] Qderiv = new double[hiddenLayerSize][inputVectorSize][inputVectorSize];
        double[][] zir = new double[hiddenLayerSize][inputVectorSize];
        double[] ui = new double[hiddenLayerSize];

        for (int i = 0; i < hiddenLayerSize; i++) {
            ui[i] = 0;
            for (int r = 0; r < inputVectorSize; r++) {
                zir[i][r] = 0;
                for (int j = 0; j < inputVectorSize; j++) {
                    zir[i][r] += correlationMatrices[i][r][j] * (inputVector[j] - centers[i][j]);
                }
                ui[i] += (zir[i][r] * zir[i][r]);
            }
        }

        for (int i = 0; i < hiddenLayerSize; i++) {
            double diffWSum = outWeights[i + 1] * diff;
            for (int j = 0; j < inputVectorSize; j++) {
                double zQsum = 0;
                for (int r = 0; r < inputVectorSize; r++) {
                    zQsum += correlationMatrices[i][r][j] * zir[i][r];
                }
                cDeriv[i][j] = 1 * expOfu(ui[i]) * diffWSum * zQsum;
            }
        }

        for (int i = 0; i < hiddenLayerSize; i++) {
            double diffWSum = outWeights[i + 1] * diff;
            for (int r = 0; r < inputVectorSize; r++) {
                for (int j = 0; j < inputVectorSize; j++) {
                    Qderiv[i][r][j] = -1 * expOfu(ui[i]) * diffWSum * (inputVector[j] - centers[i][j]) * zir[i][r];
                }
            }
        }

        for (int i = 0; i < hiddenLayerSize; i++) {
            for (int j = 0; j < inputVectorSize; j++) {
                centers[i][j] = centers[i][j] - n * cDeriv[i][j];
            }
        }

        for (int i = 0; i < hiddenLayerSize; i++) {
            for (int r = 0; r < inputVectorSize; r++) {
                //for (int j = 0; j < inputVectorSize; j++) {
                correlationMatrices[i][r][r] = correlationMatrices[i][r][r] - n * Qderiv[i][r][r];
                //}
            }
        }
    }

    //individual derivatives for testing
    public double wDeriv(int i_ind, double[] inputVector){
        double[][] zir = new double[hiddenLayerSize][inputVectorSize];
        double[] ui = new double[hiddenLayerSize];

        for (int i = 0; i < hiddenLayerSize; i++) {
            ui[i] = 0;
            for (int r = 0; r < inputVectorSize; r++) {
                zir[i][r] = 0;
                for (int j = 0; j < inputVectorSize; j++) {
                    zir[i][r] += correlationMatrices[i][r][j] * (inputVector[j] - centers[i][j]);
                }
                ui[i] += (zir[i][r] * zir[i][r]);
            }
        }

        return expOfu(ui[i_ind]);
    }

    public double centerDeriv(int i_ind, int j_ind, double[] inputVector){
        double[][] zir = new double[hiddenLayerSize][inputVectorSize];
        double[] ui = new double[hiddenLayerSize];

        for (int i = 0; i < hiddenLayerSize; i++) {
            ui[i] = 0;
            for (int r = 0; r < inputVectorSize; r++) {
                zir[i][r] = 0;
                for (int j = 0; j < inputVectorSize; j++) {
                    zir[i][r] += correlationMatrices[i][r][j] * (inputVector[j] - centers[i][j]);
                }
                ui[i] += (zir[i][r] * zir[i][r]);
            }
        }

        double zQsum = 0;
        for (int r = 0; r < inputVectorSize; r++) {
            zQsum += correlationMatrices[i_ind][r][j_ind] * zir[i_ind][r];
        }

        return 1 * expOfu(ui[i_ind]) * outWeights[i_ind + 1] * zQsum;
    }

    public double QDeriv(int i_ind, int r_ind, int j_ind, double[] inputVector){
        double[][] zir = new double[hiddenLayerSize][inputVectorSize];
        double[] ui = new double[hiddenLayerSize];

        for (int i = 0; i < hiddenLayerSize; i++) {
            ui[i] = 0;
            for (int r = 0; r < inputVectorSize; r++) {
                zir[i][r] = 0;
                for (int j = 0; j < inputVectorSize; j++) {
                    zir[i][r] += correlationMatrices[i][r][j] * (inputVector[j] - centers[i][j]);
                }
                ui[i] += (zir[i][r] * zir[i][r]);
            }
        }

        return -1 * expOfu(ui[i_ind]) * outWeights[i_ind + 1] * (inputVector[j_ind] - centers[i_ind][j_ind]) * zir[i_ind][r_ind];
    }
}
