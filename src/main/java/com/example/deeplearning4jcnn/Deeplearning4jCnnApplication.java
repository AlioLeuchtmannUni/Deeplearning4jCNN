package com.example.deeplearning4jcnn;

import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.modelimport.keras.layers.KerasFlatten;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.io.IOException;

@SpringBootApplication
public class Deeplearning4jCnnApplication {

    public Deeplearning4jCnnApplication() throws IOException {
    }


    static final WeightInit weightInit = WeightInit.XAVIER_UNIFORM;
    static final LossFunctions.LossFunction lossFunction = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;

    static final OptimizationAlgorithm optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
    static DataSetIterator mnistTrain;
    static DataSetIterator mnistTest;
    static final int batchSize = 64;
    static final int nEpochs = 20;


    static Adam getOptimizer() {

        Adam adam = new Adam();
        adam.setLearningRate(1e-3);
        adam.applySchedules(1,1);

        return adam;
    }

    static Adam createAdam() {
        Adam adam = new Adam();
        adam.setLearningRate(1e-3);
        adam.setBeta1(0.9f);
        adam.setBeta2(0.999f);
        adam.setEpsilon(1e-7f);
        //adam.applySchedules();
        return adam;
    }

    static MultiLayerNetwork createModel1(){

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(1611)
                .optimizationAlgo(optimizationAlgorithm)
                //.regularization(true)
                .learningRate(0.001)
                .updater(createAdam())
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
                .pretrain(false)
                .backprop(true)
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
                //.regularization(true)
                .learningRate(0.001)
                .updater(createAdam())
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
                .pretrain(false)
                .backprop(true)
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
                //.regularization(true)
                .learningRate(0.001)
                .updater(createAdam())
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
                .pretrain(false)
                .backprop(true)
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

        System.out.println("Evalutation Acuracy: " + eval.accuracy());
        System.out.println("Stats: " + eval.stats());

        System.out.println("Workspace size after Training and Evaluation: " + Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread().getCurrentSize());
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
    }





}
