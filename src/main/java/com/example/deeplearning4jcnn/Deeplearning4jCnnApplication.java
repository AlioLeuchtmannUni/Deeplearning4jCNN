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

import java.util.logging.Logger;


import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

@SpringBootApplication
public class Deeplearning4jCnnApplication {

    public Deeplearning4jCnnApplication() throws IOException {}
    static final Logger logger = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);
    static final WeightInit weightInit = WeightInit.XAVIER_UNIFORM;
    static final LossFunctions.LossFunction lossFunction = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;
    static final OptimizationAlgorithm optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
    static DataSetIterator mnistTrain;
    static DataSetIterator mnistTest;
    static final int batchSize = 64;
    static final int nEpochs = 20;


    /**
     * Creating {@link Adam} Optimizer with LearnRate Sheduler, updating Learnrate every epoch with: newLearnRate = learnRate * Math.pow(factor,epoch)
     * @param learnRate LearnRate
     * @param factor Factor to update Learnrate with: newLeearnRate = learnRate * Math.pow(factor,epoch)
     * @return {@link Adam}
     */
    static Adam createAdam(double learnRate,double factor) {

        Adam adam = new Adam();
        adam.setLearningRate(learnRate);
        adam.setBeta1(0.9f);
        adam.setBeta2(0.999f);
        adam.setEpsilon(1e-7f);

        Map<Integer, Double> learningRateSchedule = new HashMap<>();

        for(int epoch = 0; epoch < nEpochs ; epoch++){
            learningRateSchedule.put(epoch,learnRate * Math.pow(factor,epoch));
        }

        adam.setLearningRateSchedule(new MapSchedule(ScheduleType.EPOCH, learningRateSchedule));
        return adam;
    }

    /**
     * <pre>
     * Create Model: 784 - [24C5-P2] - 256 - 10
     * using {@link Deeplearning4jCnnApplication#optimizationAlgorithm}
     * using {@link Deeplearning4jCnnApplication#weightInit}
     * using {@link Deeplearning4jCnnApplication#lossFunction}
     * </pre>
     * @return {@link MultiLayerNetwork} created Network
     */
    static MultiLayerNetwork createModel1(){

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(1611)
                .optimizationAlgo(optimizationAlgorithm)
                .updater(createAdam(1e-3,0.95))
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                                .nIn(1)
                                .nOut(24)
                                .stride(1, 1)
                                .padding(1, 1)
                                .weightInit(weightInit)
                                .activation(Activation.RELU)
                                .build()
                )
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(2, 2)
                                .stride(2, 2)
                                .build()
                )
                .layer(2, new DenseLayer.Builder()
                                .activation(Activation.RELU)
                                .nOut(256)
                                .build()
                )
                .layer(3, new OutputLayer.Builder(lossFunction)
                                .activation(Activation.SOFTMAX)
                                .weightInit(weightInit)
                                .nOut(10)
                                .build()
                )
                .backpropType(BackpropType.Standard)
                .setInputType(InputType.convolutional(28,28,1))
                .build();

        configuration.setTrainingWorkspaceMode(WorkspaceMode.ENABLED);
        configuration.setInferenceWorkspaceMode(WorkspaceMode.ENABLED);
        return new MultiLayerNetwork(configuration);
    }

    /**
     * <pre>
     * Create Model: 784 - [24C5-P2] - [48C5-P2] - 256 - 10
     * using {@link Deeplearning4jCnnApplication#optimizationAlgorithm}
     * using {@link Deeplearning4jCnnApplication#weightInit}
     * using {@link Deeplearning4jCnnApplication#lossFunction}
     * </pre>
     * @return {@link MultiLayerNetwork} created Network
     */
    static MultiLayerNetwork createModel2(){

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(1611)
                .optimizationAlgo(optimizationAlgorithm)
                .updater(createAdam(1e-3,0.95))
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                                .nOut(24)
                                .stride(1, 1)
                                .padding(1, 1)
                                .weightInit(weightInit)
                                .activation(Activation.RELU)
                                .build()
                )
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(2, 2)
                                .stride(2, 2)
                                .build()
                )
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                                .nOut(48)
                                .stride(1, 1)
                                .padding(1, 1)
                                .weightInit(weightInit)
                                .activation(Activation.RELU)
                                .build()
                )
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(2, 2)
                                .stride(2, 2)
                                .build()
                )
                .layer(4, new DenseLayer.Builder()
                                .activation(Activation.RELU)
                                .nOut(256)
                                .build()
                )
                .layer(5, new OutputLayer.Builder(lossFunction)
                                .activation(Activation.SOFTMAX)
                                .weightInit(weightInit)
                                .nOut(10)
                                .build()
                )
                .backpropType(BackpropType.Standard)
                .setInputType(InputType.convolutional(28,28,1))
                .build();

        configuration.setTrainingWorkspaceMode(WorkspaceMode.ENABLED);
        configuration.setInferenceWorkspaceMode(WorkspaceMode.ENABLED);
        return new MultiLayerNetwork(configuration);
    }

    /**
     * <pre>
     * Create Model: 784 - [24C5-P2] - [48C5-P2] - [64C5-P2] - 256 - 10
     * using {@link Deeplearning4jCnnApplication#optimizationAlgorithm}
     * using {@link Deeplearning4jCnnApplication#weightInit}
     * using {@link Deeplearning4jCnnApplication#lossFunction}
     * </pre>
     * @return {@link MultiLayerNetwork} created Network
     */
    static MultiLayerNetwork createModel3(){

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(1611)
                .optimizationAlgo(optimizationAlgorithm)
                .updater(createAdam(1e-3,0.95))
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                                .nIn(1)
                                .nOut(24)
                                .stride(1, 1)
                                .padding(1, 1)
                                .weightInit(weightInit)
                                .activation(Activation.RELU)
                                .build()
                )
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(2, 2)
                                .stride(2, 2)
                                .build()
                ).layer(2,
                        new ConvolutionLayer.Builder(5, 5)
                                .nOut(48)
                                .stride(1, 1)
                                .padding(1, 1)
                                .weightInit(weightInit)
                                .activation(Activation.RELU)
                                .build()
                ).layer(3,
                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(2, 2)
                                .stride(2, 2)
                                .build()
                ).layer(4,
                        new ConvolutionLayer.Builder(5, 5)
                                .nOut(64)
                                .stride(1, 1)
                                .padding(1, 1)
                                .weightInit(weightInit)
                                .activation(Activation.RELU)
                                .build()
                ).layer(5,
                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(2, 2)
                                .stride(2, 2)
                                .build()
                ).layer(6,
                        new DenseLayer.Builder()
                                .activation(Activation.RELU)
                                .nOut(256)
                                .build()
                ).layer(7,
                        new OutputLayer.Builder(lossFunction)
                                .activation(Activation.SOFTMAX)
                                .weightInit(weightInit)
                                .nOut(10)
                                .build()
                )
                .backpropType(BackpropType.Standard)
                .setInputType(InputType.convolutional(28,28,1))
                .build();

        configuration.setTrainingWorkspaceMode(WorkspaceMode.ENABLED);
        configuration.setInferenceWorkspaceMode(WorkspaceMode.ENABLED);
        return new MultiLayerNetwork(configuration);
    }

    public static void main(String[] args) throws IOException {
        SpringApplication.run(Deeplearning4jCnnApplication.class, args);

        logger.info("Start");
        Nd4j.getMemoryManager().setAutoGcWindow(5000);

        MnistDataset mnistDataset = new MnistDataset("trainingSet/","./src/main/resources/",batchSize);
        mnistTrain = mnistDataset.getTrainDataset();
        mnistTest = mnistDataset.getTestDataset();

        trainAndEvalModel(createModel1());
        trainAndEvalModel(createModel2());
        trainAndEvalModel(createModel3());
        logger.info("Done");

    }


    /**
     * <pre>
     *     Initializing Network Training for  {@link Deeplearning4jCnnApplication#nEpochs} and printing evaluation for every Epoch.
     * </pre>
     * @param network {@link MultiLayerNetwork} Network to Train and Evaluate
     */
    static void trainAndEvalModel(MultiLayerNetwork network){

        network.init();

        for(int i = 0; i < nEpochs; i++){
            logger.info("Process Epoch "+(i+1)+" ...");
            network.fit(mnistTrain);

            Evaluation eval = new Evaluation(10);
            network.doEvaluation(mnistTest,eval);
            logger.info("Accuracy for previous Epoch: " + eval.accuracy());

        }

        Evaluation eval = new Evaluation(10);
        network.doEvaluation(mnistTest,eval);

        logger.info(" \n Final Accuracy: " + eval.accuracy() + "");
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
    }




}
