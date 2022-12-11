package com.example.deeplearning4jcnn;

import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;


import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

@SpringBootApplication
public class Deeplearning4jCnnApplication {

    public Deeplearning4jCnnApplication() throws IOException {
    }


    static final WeightInit weightInit = WeightInit.XAVIER_UNIFORM;
    // https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81#:~:text=Negative%20log%2Dlikelihood%20minimization%20is,up%20the%20correct%20log%20probabilities.%E2%80%9D
    // use next best available
    static final LossFunctions.LossFunction lossFunction = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;
    static final OptimizationAlgorithm optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
    static DataSetIterator mnistTrain;
    static DataSetIterator mnistTest;
    static final int batchSize = 64;
    static final int nEpochs = 20;



    //  <artifactId>nd4j-api</artifactId> import nötig
    // https://github.com/deeplearning4j/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/convolution/LeNetMNISTReLu.java#L116-L128
    // https://deeplearning4j.konduit.ai/deeplearning4j/reference/updaters-optimizers
    static Adam createAdam(double factor) {

        double startLEarningRate = 1e-3;
        Adam adam = new Adam();
        adam.setLearningRate(1e-3);
        adam.setBeta1(0.9f);
        adam.setBeta2(0.999f);
        adam.setEpsilon(1e-7f);

        Map<Integer, Double> learningRateSchedule = new HashMap<>();

        for(int epoch = 0; epoch < nEpochs ; epoch++){
            double newLearningRate = startLEarningRate * Math.pow(factor,epoch);
            learningRateSchedule.put(epoch,newLearningRate);
        }


        //adam.setLearningRateSchedule(new MapSchedule(ScheduleType.EPOCH, learningRateSchedule));
        return adam;
    }

    static MultiLayerNetwork createModel1(){

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(1611)
                .optimizationAlgo(optimizationAlgorithm)
                .updater(createAdam(0.95))
                .list()
                .layer(0,
                        new ConvolutionLayer.Builder(5, 5)
                                .nIn(1)
                                .nOut(24)
                                .stride(1, 1)
                                .padding(1, 1)
                                .weightInit(weightInit)
                                .activation(Activation.RELU)
                                .build())
                .layer(1,
                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(2, 2)
                                .stride(2, 2)
                                .build())
                .layer(2,
                        new DenseLayer.Builder()
                                .activation(Activation.RELU)
                                .nOut(256)
                                .build())
                .layer(3,
                        new OutputLayer.Builder(lossFunction)
                                .activation(Activation.SOFTMAX)
                                .weightInit(weightInit)
                                .nOut(10)
                                .build())
                .backpropType(BackpropType.Standard)
                .setInputType(InputType.convolutional(28,28,1))
                .build();
        configuration.setTrainingWorkspaceMode(WorkspaceMode.SEPARATE);
        configuration.setInferenceWorkspaceMode(WorkspaceMode.SINGLE);
        return new MultiLayerNetwork(configuration);
    }

    static MultiLayerNetwork createModel2(){

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(1611)
                .optimizationAlgo(optimizationAlgorithm)
                .updater(createAdam(0.95))
                .list()
                .layer(0,
                        new ConvolutionLayer.Builder(5, 5)
                                .nOut(24)
                                .stride(1, 1)
                                .padding(1, 1)
                                .weightInit(weightInit)
                                .activation(Activation.RELU)
                                .build())
                .layer(1,
                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(2, 2)
                                .stride(2, 2)
                                .build())
                .layer(2,
                        new ConvolutionLayer.Builder(5, 5)
                                .nOut(48)
                                .stride(1, 1)
                                .padding(1, 1)
                                .weightInit(weightInit)
                                .activation(Activation.RELU)
                                .build())
                .layer(3,
                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(2, 2)
                                .stride(2, 2)
                                .build())
                .layer(4,
                        new DenseLayer.Builder()
                                .activation(Activation.RELU)
                                .nOut(256)
                                .build())
                .layer(5,
                        new OutputLayer.Builder(lossFunction)
                                .activation(Activation.SOFTMAX)
                                .weightInit(weightInit)
                                .nOut(10)
                                .build())
                .backpropType(BackpropType.Standard)
                .setInputType(InputType.convolutional(28,28,1))
                .build();

        configuration.setTrainingWorkspaceMode(WorkspaceMode.SEPARATE);
        configuration.setInferenceWorkspaceMode(WorkspaceMode.SINGLE);
        return new MultiLayerNetwork(configuration);
    }

    static MultiLayerNetwork createModel3(){

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(1611)
                .optimizationAlgo(optimizationAlgorithm)
                .updater(createAdam(0.95))
                .list()
                .layer(0,
                        new ConvolutionLayer.Builder(5, 5)
                                .nIn(1)
                                .nOut(24)
                                .stride(1, 1)
                                .padding(1, 1)
                                .weightInit(weightInit)
                                .activation(Activation.RELU)
                                .build())
                .layer(1,
                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(2, 2)
                                .stride(2, 2)
                                .build())
                .layer(2,
                        new ConvolutionLayer.Builder(5, 5)
                                .nOut(48)
                                .stride(1, 1)
                                .padding(1, 1)
                                .weightInit(weightInit)
                                .activation(Activation.RELU)
                                .build())
                .layer(3,
                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(2, 2)
                                .stride(2, 2)
                                .build())
                .layer(4,
                        new ConvolutionLayer.Builder(5, 5)
                                .nOut(64)
                                .stride(1, 1)
                                .padding(1, 1)
                                .weightInit(weightInit)
                                .activation(Activation.RELU)
                                .build())
                .layer(5,
                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(2, 2)
                                .stride(2, 2)
                                .build())
                .layer(6,
                        new DenseLayer.Builder()
                                .activation(Activation.RELU)
                                .nOut(256)
                                .build())
                .layer(7,
                        new OutputLayer.Builder(lossFunction)
                                .activation(Activation.SOFTMAX)
                                .weightInit(weightInit)
                                .nOut(10)
                                .build())
                .backpropType(BackpropType.Standard)
                .setInputType(InputType.convolutional(28,28,1))
                .build();

        configuration.setTrainingWorkspaceMode(WorkspaceMode.SEPARATE);
        configuration.setInferenceWorkspaceMode(WorkspaceMode.SINGLE);
        return new MultiLayerNetwork(configuration);
    }

    public static void main(String[] args) throws IOException {
        SpringApplication.run(Deeplearning4jCnnApplication.class, args);

        Nd4j.getMemoryManager().setAutoGcWindow(5000);

        MnistDataset mnistDataset = new MnistDataset(batchSize); //initialisierung statischer Variablen
        mnistTrain = MnistDataset.getTrainDataset();
        mnistTest = MnistDataset.getTestDataset();

        trainAndEvalModel(createModel1());
        trainAndEvalModel(createModel2());
        trainAndEvalModel(createModel3());

    }


    static void trainAndEvalModel(MultiLayerNetwork network){

        System.out.println("Workspace size before Network init: " + Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread().getCurrentSize());
        network.init();


        for(int i = 0; i < nEpochs; i++){
            System.out.println("Process Epoch "+(i+1)+" ...");
            network.fit(mnistTrain);

            Evaluation eval = new Evaluation(10);
            network.doEvaluation(mnistTest,eval);
            System.out.println("Accuracy for previous Epoch: " + eval.accuracy());

        }

        Evaluation eval = new Evaluation(10);
        network.doEvaluation(mnistTest,eval);

        System.out.println(" \n Final Accuracy: " + eval.accuracy() + "");
        // uncomment for extended Information
        //System.out.println("Stats: " + eval.stats() + " \n");

        System.out.println("Workspace size after Training and Evaluation: " + Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread().getCurrentSize());
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
    }




}
