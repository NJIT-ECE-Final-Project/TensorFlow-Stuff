package com.njit.ece.senior_project.tensorflow;

import com.njit.ece.senior_project.data.SampleLoader.SampleDataLoader;

import java.io.File;

/**
 * Tests the activity classifier
 */
public class ActivityClassifierTest {

    public static void main(String[] args) throws Exception {

        // create a classifier
        ActivityClassifier classifier = new ActivityClassifier();

        // load sample data
        File sampleDataFile = new File(LoadModelTest.class.getResource("data/UCI HAR Dataset/test/Inertial Signals").toURI());
        SampleDataLoader dataLoader = new SampleDataLoader(sampleDataFile);

        // run the classifier for each test set
        while(dataLoader.hasNext()) {
            float[][][] dataSet = dataLoader.getNextRow();

            // run the data through the classifier
            ActivityClassifier.Activity activity = classifier.classify(dataSet);

            System.out.println("The activity is: " + activity + " (" + activity.getNum() + ")");
        }
    }

}
