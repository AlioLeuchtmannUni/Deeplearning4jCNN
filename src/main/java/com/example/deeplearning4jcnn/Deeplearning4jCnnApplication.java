package com.example.deeplearning4jcnn;

import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.modelimport.keras.layers.KerasFlatten;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
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

    static DataSetIterator mnistTrain;
    static DataSetIterator mnistTest;
    static int batchSize = 64;
    static int nEpochs = 10;


    static Adam getOptimizer() {

        Adam adam = new Adam();
        adam.setLearningRate(1e-3);
        adam.applySchedules(1,1);

        return adam;
    }

    static MultiLayerNetwork createModel1(){

        Adam adam = new Adam();
        adam.setLearningRate(1e-3);

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(1611)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                //.regularization(true)
                .learningRate(0.001)
                .updater(adam)
                .list()
                .layer(0,
                        new ConvolutionLayer.Builder(5, 5)
                                .nIn(1)
                                .nOut(24)
                                .stride(1, 1)
                                .padding(1, 1)
                                .weightInit(WeightInit.XAVIER_UNIFORM)
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
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .activation(Activation.SOFTMAX)
                                .weightInit(WeightInit.XAVIER_UNIFORM)
                                .nOut(10)
                                .build())
                .pretrain(false)
                .backprop(true)
                .setInputType(InputType.convolutional(28,28,1))
                .build();


        return new MultiLayerNetwork(configuration);
    }

    static MultiLayerNetwork createModel2(){

        Adam adam = new Adam();
        adam.setLearningRate(1e-3);

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(1611)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                //.regularization(true)
                .learningRate(0.001)
                .updater(adam)
                .list()
                .layer(0,
                        new ConvolutionLayer.Builder(5, 5)
                                .nOut(24)
                                .stride(1, 1)
                                .padding(1, 1)
                                .weightInit(WeightInit.XAVIER_UNIFORM)
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
                                .weightInit(WeightInit.XAVIER_UNIFORM)
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
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .activation(Activation.SOFTMAX)
                                .weightInit(WeightInit.XAVIER_UNIFORM)
                                .nOut(10)
                                .build())
                .pretrain(false)
                .backprop(true)
                .setInputType(InputType.convolutional(28,28,1))
                .build();

        return new MultiLayerNetwork(configuration);
    }

    static MultiLayerNetwork createModel3(){

        Adam adam = new Adam();
        adam.setLearningRate(1e-3);

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(1611)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                //.regularization(true)
                .learningRate(0.001)
                .updater(adam)
                .list()
                .layer(0,
                        new ConvolutionLayer.Builder(5, 5)
                                .nIn(1)
                                .nOut(24)
                                .stride(1, 1)
                                .padding(1, 1)
                                .weightInit(WeightInit.XAVIER_UNIFORM)
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
                                .weightInit(WeightInit.XAVIER_UNIFORM)
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
                                .weightInit(WeightInit.XAVIER_UNIFORM)
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
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .activation(Activation.SOFTMAX)
                                .weightInit(WeightInit.XAVIER_UNIFORM)
                                .nOut(10)
                                .build())
                .pretrain(false)
                .backprop(true)
                .setInputType(InputType.convolutional(28,28,1))
                .build();

        return new MultiLayerNetwork(configuration);
    }

    public static void main(String[] args) throws IOException {
        SpringApplication.run(Deeplearning4jCnnApplication.class, args);

        MnistDataset mnistDataset = new MnistDataset();
        mnistTrain = MnistDataset.getTrainDataset();
        mnistTest = MnistDataset.getTestDataset();


        trainAndEvalModel(createModel1());
        trainAndEvalModel(createModel2());
        trainAndEvalModel(createModel3());

    }


    static void trainAndEvalModel(MultiLayerNetwork network){

        network.init();

        network.setListeners(new ScoreIterationListener(100));

        for(int i=0; i < nEpochs; i++){
            System.out.println("Process Epoch "+(i+1)+" ...");
            network.fit(mnistTrain);
        }

        Evaluation eval = new Evaluation(10);

        while(mnistTest.hasNext()){
            DataSet next = mnistTest.next();
            INDArray output = network.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        System.out.println(eval.accuracy());
    }





}
