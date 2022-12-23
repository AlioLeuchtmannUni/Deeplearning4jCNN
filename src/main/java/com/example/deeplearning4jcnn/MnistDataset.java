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

public class MnistDataset {

    private final int height = 28;
    private final int width = 28;
    private final int channels = 1;
    private final int batchSize; // Minibatch size. Here: The number of images to fetch for each call to dataIter.next().
    private final int labelIndex = 1; // Index of the label Writable (usually an IntWritable), as obtained by recordReader.next()
    public int trainingSetSize;
    public int testSetSize;
    private final InputSplit trainData;
    private final InputSplit testData;
    private final ParentPathLabelGenerator labelMaker;

    /**
     * <pre>
     * Requirements for mnist Example:
     *     // 1. Download https://www.kaggle.com/datasets/scolianni/mnistasjpg
     *     // 2. Extract Files
     *     // 3. put Folder trainingSet in Resources Folder
     *
     *
     * Using
     * </pre>
     * @param batchSize BatchSize
     * @param folderName Name of the Folder containing the Trainingset divided in Folders by label
     * @param dataLocalPath Path to Folder containing the Data
     */
    MnistDataset(String folderName, String dataLocalPath,int batchSize){

        this.batchSize = batchSize;

        File parentDir = new File(dataLocalPath,folderName);
        //Images are of format given by allowedExtension -
        String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
        long seed = 123;
        Random randNumGen = new Random(seed);
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);
        this.labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, this.labelMaker);

        //Split the image files into train and test. Specify the train test split as 80%,20%
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);

        this.trainData = filesInDirSplit[0];
        this.testData = filesInDirSplit[1];
        System.out.println("data split");
    }

    /**
     * Creating TrainingDataset from {@link MnistDataset#trainData} initialized in Constructor
     * @return {@link DataSetIterator} TrainingDataSet
     */
    public DataSetIterator getTrainDataset() throws IOException {

        ImageRecordReader recordReader = new ImageRecordReader(this.height,this.width,this.channels,this.labelMaker);
        recordReader.initialize(this.trainData);
        int outputNum = recordReader.numLabels();
        this.trainingSetSize = outputNum;
        System.out.println("initialized Training set, "+outputNum+" Labels");
        return new RecordReaderDataSetIterator(recordReader, this.batchSize, this.labelIndex, outputNum);
    }


    /**
     * Creating TrainingDataset from {@link MnistDataset#testData} initialized in Constructor
     * @return {@link DataSetIterator} TrainingDataSet
     */
    public DataSetIterator getTestDataset() throws IOException {

        ImageRecordReader recordReader = new ImageRecordReader(this.height,this.width,this.channels,this.labelMaker);
        recordReader.initialize(testData);
        int outputNum = recordReader.numLabels();
        this.testSetSize = outputNum;
        System.out.println("initialized Test set, "+outputNum+" Labels");
        return new RecordReaderDataSetIterator(recordReader, this.batchSize, this.labelIndex, outputNum);
    }





}
