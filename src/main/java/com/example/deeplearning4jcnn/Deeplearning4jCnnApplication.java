package com.example.deeplearning4jcnn;

import org.deeplearning4j.clustering.strategy.OptimisationStrategy;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.InputTypeUtil;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.modelimport.keras.layers.KerasFlatten;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.io.IOException;

@SpringBootApplication
public class Deeplearning4jCnnApplication {

    public Deeplearning4jCnnApplication() throws IOException {
    }

    public static void main(String[] args) throws IOException {
        SpringApplication.run(Deeplearning4jCnnApplication.class, args);

        int batchSize = 64;
        int nEpochs = 1;
        //DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true,12345);
        //DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false,12345);

        Adam adam = new Adam();
        adam.setLearningRate(1e-3);


        MultiLayerConfiguration configurationModel1 = new NeuralNetConfiguration.Builder()
                .seed(1611)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.001)
                //.regularization(true)
                .updater(adam)
                .list()
                .layer(0,
                        new ConvolutionLayer.Builder(5, 5)
                                .nIn(3)
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
                        new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                                .activation(Activation.RELU)
                                .weightInit(WeightInit.XAVIER_UNIFORM)
                                .nOut(256)
                                .build())
                .layer(3,
                        new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                                .activation(Activation.SOFTMAX)
                                .weightInit(WeightInit.XAVIER_UNIFORM)
                                .nOut(10)
                                .build())
                .pretrain(false)
                .backprop(true)
                .setInputType(InputType.convolutional(28,28,1))
                .build();


        MultiLayerNetwork network1 = new MultiLayerNetwork(configurationModel1);
        network1.init();
    }





}
