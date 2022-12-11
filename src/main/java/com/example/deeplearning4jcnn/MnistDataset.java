package com.example.deeplearning4jcnn;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.ShowImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/*
CSV Mit Format:

label,pixel,...,pixel\n

* */

public class MnistDataset {

    //Images are of format given by allowedExtension -
    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    private static final long seed = 123;

    private static final Random randNumGen = new Random(seed);

    private static final int height = 28;
    private static final int width = 28;
    private static final int channels = 1;

    private static int batchSize; // Minibatch size. Here: The number of images to fetch for each call to dataIter.next().
    private static final int labelIndex = 1; // Index of the label Writable (usually an IntWritable), as obtained by recordReader.next()

    public static String dataLocalPath = "./src/main/resources/";

    public static int trainingSetSize;
    public static int testSetSize;

    private static InputSplit trainData;
    private static InputSplit testData;

    private static ParentPathLabelGenerator labelMaker;


    // 1. Download https://www.kaggle.com/datasets/scolianni/mnistasjpg
    // 2. Extract Files
    // 3. put Folder trainingSet in Resources Folder
    MnistDataset(int batchSize){

        MnistDataset.batchSize = batchSize;

        File parentDir = new File(dataLocalPath,"trainingSet/");
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);
        labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);

        //Split the image files into train and test. Specify the train test split as 80%,20%
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);


        trainData = filesInDirSplit[0];
        testData = filesInDirSplit[1];
        System.out.println("data split");
    }

    public static DataSetIterator getTrainDataset() throws IOException {

        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        recordReader.initialize(trainData);
        int outputNum = recordReader.numLabels();
        trainingSetSize = outputNum;
        System.out.println("initialized Training set, "+outputNum+" Labels");
        return new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, outputNum);
    }

    public static DataSetIterator getTestDataset() throws IOException {

        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        recordReader.initialize(testData);
        int outputNum = recordReader.numLabels();
        testSetSize = outputNum;
        System.out.println("initialized Test set, "+outputNum+" Labels");
        return new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, outputNum);
    }





}
