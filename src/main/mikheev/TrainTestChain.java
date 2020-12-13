package mikheev;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class TrainTestChain {

    //network
    private RadialBasedNetwork radialBasedNetwork;
    //dataset
    private Dataset dataset = Dataset.MONEY;
    //normalizeOrScale
    private boolean normalizeVectorsSquaredSum = false;
    private boolean scaleToFrom0To1 = false;
    //trainTestSplitArgs
    private boolean percentSplit = true;
    private boolean exactSplit = false;
    private double trainValuePercent = 0.7;
    private int exactSplitIndex;
    //centersInitializationType
    private boolean minMaxInit = true;
    private boolean importantPointsInit = false;
    private boolean randomInBoundsInit = false;
    //networkParameters
    private int inputVectorSize = 5;
    private int hiddenLayerSize = 10;
    private boolean basicRadial = true;
    private boolean hyperRadial = false;
    private boolean initWithKMeans = true;
    private int maxEpochForKmeans = 100;
    private int maxEpochForBackprop = 1000;
    private double nForKmeans = 0.01;
    private double nForBackprop = 0.01;
    private double shiftStop = 0.001;
    //errorTrend
    private List<Double> lastBackpropErrors;
    //testNormalizationParameter
    private boolean renormalize = false;

    public class Builder {
        TrainTestChain trainTestChain;

        public Builder() {
            trainTestChain = new TrainTestChain();
        }

        private TrainTestChain build() {
            return trainTestChain;
        }

        public Builder setDataset(Dataset dataset){
            trainTestChain.dataset = dataset;
            return this;
        }

        public Builder setPercentSplit(double trainValuePercent) {
            trainTestChain.percentSplit = true;
            trainTestChain.exactSplit = false;
            trainTestChain.trainValuePercent = trainValuePercent;
            return this;
        }

        public Builder setExactSplit(int exactSplitIndex) {
            trainTestChain.percentSplit = false;
            trainTestChain.exactSplit = true;
            trainTestChain.exactSplitIndex = exactSplitIndex;
            return this;
        }

        public Builder enableNormalization() {
            trainTestChain.normalizeVectorsSquaredSum = true;
            trainTestChain.scaleToFrom0To1 = false;
            return this;
        }

        public Builder enableScaling() {
            trainTestChain.normalizeVectorsSquaredSum = false;
            trainTestChain.scaleToFrom0To1 = true;
            return this;
        }

        public Builder enableMinMaxInit() {
            trainTestChain.minMaxInit = true;
            trainTestChain.importantPointsInit = false;
            trainTestChain.randomInBoundsInit = false;
            return this;
        }

        public Builder enableImportantPointsInit() {
            trainTestChain.minMaxInit = false;
            trainTestChain.importantPointsInit = true;
            trainTestChain.randomInBoundsInit = false;
            return this;
        }

        public Builder enableRandomInBoundsInit() {
            trainTestChain.minMaxInit = false;
            trainTestChain.importantPointsInit = false;
            trainTestChain.randomInBoundsInit = true;
            return this;
        }

        public Builder setInputVectorSize(int size) {
            trainTestChain.inputVectorSize = size;
            return this;
        }

        public Builder setHiddenLayerSize(int size) {
            trainTestChain.hiddenLayerSize = size;
            return this;
        }

        public Builder pickBasicRadial() {
            trainTestChain.basicRadial = true;
            trainTestChain.hyperRadial = false;
            return this;
        }

        public Builder pickHyperRadial() {
            trainTestChain.hyperRadial = false;
            trainTestChain.basicRadial = true;
            return this;
        }

        public Builder enableKmeansInit() {
            trainTestChain.initWithKMeans = true;
            return this;
        }

        public Builder disableKmeansInit() {
            trainTestChain.initWithKMeans = false;
            return this;
        }

        public Builder setMaxEpochForKmeans(int maxEpoch) {
            trainTestChain.maxEpochForKmeans = maxEpoch;
            return this;
        }

        public Builder setShiftStopForKmeans(double shiftStop) {
            trainTestChain.shiftStop = shiftStop;
            return this;
        }

        public Builder setNForKmeans(double n) {
            trainTestChain.nForKmeans = n;
            return this;
        }

        public Builder setMaxEpochForBackprop(int maxEpoch) {
            trainTestChain.maxEpochForBackprop = maxEpoch;
            return this;
        }

        public Builder setNForBackprop(int nForBackprop) {
            trainTestChain.nForBackprop = nForBackprop;
            return this;
        }
    }

    void trainNetwork(){
        //buildTrainTestData
        double[] inputValues = Arrays.copyOf(dataset.getInputData(), dataset.getInputData().length);
        int totalVectors = inputValues.length - inputVectorSize;
        double[][] inputVectors = new double[totalVectors][inputVectorSize];
        double[] expectedOutputValues = new double[totalVectors];
        for (int i = 0; i < totalVectors; i++) {
            for (int j = 0; j < inputVectorSize; j++) {
                inputVectors[i][j] = inputValues[i + j];
            }
            expectedOutputValues[i] = inputValues[i + inputVectorSize];
        }

        int trainSampleSize = 0;
        int testSampleSize = 0;
        if (percentSplit) {
            trainSampleSize = (int)(trainValuePercent * totalVectors);
        }
        if (exactSplit) {
            trainSampleSize = exactSplitIndex - inputVectorSize;
        }
        testSampleSize = totalVectors - trainSampleSize;

        //normalize if chosen to
        double[] norm_sums = new double[totalVectors];
        if (normalizeVectorsSquaredSum){
            for (int i = 0; i < totalVectors; i++) {
                double norm_sum = 0;
                for (int j = 0; j < inputVectorSize; j++) {
                    norm_sum += Math.pow(inputVectors[i][j], 2);
                }
                norm_sum += Math.pow(expectedOutputValues[i], 2);
                norm_sum = Math.sqrt(norm_sum);
                for (int j = 0; j < inputVectorSize; j++) {
                    inputVectors[i][j] /= norm_sum;
                }
                expectedOutputValues[i] /= norm_sum;
                norm_sums[i] = norm_sum;
            }
        }

        //extractTrain
        double[][] trainInputVectors = new double[trainSampleSize][inputVectorSize];
        double[] trainExpectedOutputValues = new double[trainSampleSize];
        for (int i = 0; i < trainSampleSize; i++) {
            trainInputVectors[i] = inputVectors[i];
            trainExpectedOutputValues[i] = expectedOutputValues[i];
        }

        //buildInitializationSet
        double[][] initInputVects = new double[hiddenLayerSize][inputVectorSize];

        if (minMaxInit) {
            List<Integer> indexes = new ArrayList<>();
            int sections = hiddenLayerSize / 2 + hiddenLayerSize % 2;
            int maxTotal = trainSampleSize - inputVectorSize;
            int sectionSize = maxTotal / sections;
            for (int i = 0; i < sections; i++) {
                int sectionMinI = sectionSize * i;
                int sectionMaxI = sectionSize * i;
                for (int j = 0; j < sectionSize; j++) {
                    if (dataset.getInputData()[sectionMinI] > dataset.getInputData()[j + i * sectionSize]) {
                        sectionMinI = j + i * sectionSize;
                    }
                    if (dataset.getInputData()[sectionMinI] < dataset.getInputData()[j + i * sectionSize]) {
                        sectionMaxI = j + i * sectionSize;
                    }
                }
                //work on this, there is bug
                int indexOfStartMin = sectionMinI - inputVectorSize / 2;
                int indexOfStartMax = sectionMaxI + inputVectorSize / 2;
                if (indexOfStartMin < 0) indexOfStartMin = 0;
                if (indexOfStartMax >= testSampleSize) indexOfStartMax = testSampleSize - 1;
                indexes.add(indexOfStartMin);
                indexes.add(indexOfStartMax);
            }
            for (int i = 0; i < hiddenLayerSize; i++) {
                initInputVects[i] = trainInputVectors[indexes.get(i)];
            }
        }
        if (importantPointsInit) {
            //implement
        }
        if (randomInBoundsInit) {
            //implement
        }

        //create and train Network
        if (basicRadial){
            radialBasedNetwork = new RBNbasic(inputVectorSize, hiddenLayerSize);
        }

        if (hyperRadial){

        }

        if (initWithKMeans){
            radialBasedNetwork.initCentersKmeans(initInputVects, trainInputVectors, maxEpochForKmeans, nForKmeans, shiftStop);
        } else {
            radialBasedNetwork.initCentersNoKmeans(initInputVects);
        }
        radialBasedNetwork.initW();
        lastBackpropErrors = radialBasedNetwork
                .trainWonly(trainInputVectors, trainExpectedOutputValues, maxEpochForBackprop, nForBackprop, 0.01);
    }

    public TestResuts testNetwork() {
        double[] inputValues = Arrays.copyOf(dataset.getInputData(), dataset.getInputData().length);
        int totalVectors = inputValues.length - inputVectorSize;
        double[][] inputVectors = new double[totalVectors][inputVectorSize];
        double[] expectedOutputValues = new double[totalVectors];
        for (int i = 0; i < totalVectors; i++) {
            for (int j = 0; j < inputVectorSize; j++) {
                inputVectors[i][j] = inputValues[i + j];
            }
            expectedOutputValues[i] = inputValues[i + inputVectorSize];
        }

        int trainSampleSize = 0;
        int testSampleSize = 0;
        if (percentSplit) {
            trainSampleSize = (int)(trainValuePercent * totalVectors);
        }
        if (exactSplit) {
            trainSampleSize = exactSplitIndex - inputVectorSize;
        }
        testSampleSize = totalVectors - trainSampleSize;

        //normalize if chosen to
        double[] norm_sums = new double[totalVectors];
        if (normalizeVectorsSquaredSum){
            for (int i = 0; i < totalVectors; i++) {
                double norm_sum = 0;
                for (int j = 0; j < inputVectorSize; j++) {
                    norm_sum += Math.pow(inputVectors[i][j], 2);
                }
                norm_sum += Math.pow(expectedOutputValues[i], 2);
                norm_sum = Math.sqrt(norm_sum);
                for (int j = 0; j < inputVectorSize; j++) {
                    inputVectors[i][j] /= norm_sum;
                }
                expectedOutputValues[i] /= norm_sum;
                norm_sums[i] = norm_sum;
            }
        }

        double[] normalizedResults = new double[totalVectors];
        for (int i = 0; i < totalVectors; i++) {
            normalizedResults[i] = radialBasedNetwork.f(inputVectors[i]);
        }
        double[] denormalizedResults = new double[totalVectors];
        //denormalizeIfWasEnabled
        if (normalizeVectorsSquaredSum) {
            for (int i = 0; i < totalVectors; i++) {
                denormalizedResults[i] = normalizedResults[i] * norm_sums[i];
            }
        } else {
            for (int i = 0; i < totalVectors; i++) {
                denormalizedResults[i] = normalizedResults[i];
            }
        }

        //trendPrediction
        double[] trend = new double[testSampleSize + 1];
        double[] denormalizedTrend = new double[testSampleSize + 1];
        double[] currentInput = new double[inputVectorSize];
        for (int i = 0; i < inputVectorSize; i++) {
            currentInput[i] = inputVectors[trainSampleSize - 1][i];
        }
        double current_norm_sum = norm_sums[trainSampleSize - 1];
        for (int i = trainSampleSize - 1; i < expectedOutputValues.length; i++) {
            double res = radialBasedNetwork.f(currentInput);
            trend[i - (trainSampleSize - 1)] = res;
            if (normalizeVectorsSquaredSum) {
                denormalizedTrend[i - (trainSampleSize - 1)] = res * current_norm_sum;
                double new_norm_sum = 0;
                for (int j = 0; j < inputVectorSize; j++) {
                    new_norm_sum += Math.pow(currentInput[j] * current_norm_sum, 2);
                }
                new_norm_sum += Math.pow(res * current_norm_sum, 2);
                new_norm_sum = Math.sqrt(new_norm_sum);
                for (int j = 0; j < inputVectorSize - 1; j++) {
                    currentInput[j] = currentInput[j + 1];
                }
                currentInput[inputVectorSize - 1] = res;
                if (renormalize) {
                    for (int j = 0; j < inputVectorSize; j++) {
                        currentInput[j] = currentInput[j] * current_norm_sum;
                        currentInput[j] = currentInput[j] / new_norm_sum;
                    }
                    current_norm_sum = new_norm_sum;
                }
            } else {
                for (int j = 0; j < inputVectorSize - 1; j++) {
                    currentInput[j] = currentInput[j + 1];
                }
                currentInput[inputVectorSize - 1] = res;
            }
        }
        TestResuts testResuts = new TestResuts();
        testResuts.testSplitIndex = trainSampleSize;
        testResuts.normalizedResults = normalizedResults;
        testResuts.denormalizedResults = denormalizedResults;
        testResuts.trend = trend;
        testResuts.denormalizedTrend = denormalizedTrend;
        return testResuts;
    }

    public static class TestResuts {
        public int testSplitIndex;
        public double[] normalizedResults;
        public double[] denormalizedResults;
        public double[] trend;
        public double[] denormalizedTrend;
    }

    public void serializeNetworkToFile(String filename) throws IOException {
        ObjectOutputStream serializationStream = new ObjectOutputStream(new FileOutputStream(filename));
        serializationStream.writeObject(radialBasedNetwork);
        serializationStream.flush();
    }

    public void loadNetworkFromFile(String filename) throws IOException, ClassNotFoundException {
        ObjectInputStream serializationStream = new ObjectInputStream(new FileInputStream(filename));
        radialBasedNetwork = (RadialBasedNetwork) serializationStream.readObject();
    }
}
