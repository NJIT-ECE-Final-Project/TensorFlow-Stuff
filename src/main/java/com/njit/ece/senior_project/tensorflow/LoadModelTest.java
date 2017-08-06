package com.njit.ece.senior_project.tensorflow;

import com.njit.ece.senior_project.data.SampleLoader.SampleDataLoader;
import com.njit.ece.senior_project.data.util.ArrayHelper;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.FloatBuffer;

/**
 * Test loading the model and passing in some data
 */
public class LoadModelTest {


    public static void main(String[] args) throws IOException, URISyntaxException {
        // good idea to print the version number, 1.2.0 as of this writing
        System.out.println(TensorFlow.version());

        final long[] X_SHAPE = new long[]{1, 128,9};

        float[] x_vals = new float[(int)X_SHAPE[0]*(int)X_SHAPE[1]*(int)X_SHAPE[2]];

        for(int i = 0; i < x_vals.length; i++) {
            if(i % 6 == 0) {
                x_vals[i] = -1.0f;
            } else {
                x_vals[i] = 0.0f;
            }
        }

        File sampleDataFile = new File(LoadModelTest.class.getResource("data/UCI HAR Dataset/test/Inertial Signals").toURI());
        SampleDataLoader dataLoader = new SampleDataLoader(sampleDataFile);
        float[][][] dataSet = dataLoader.loadOneRowAllSignals();




        File trainedModelFile = new File(LoadModelTest.class.getResource("60").getFile());


        // load the model Bundle and run the data
        try (SavedModelBundle b = SavedModelBundle.load(trainedModelFile.getAbsolutePath(), "serve")) {

            // create the session from the Bundle
            Session sess = b.session();
            // create an input Tensor, value = 2.0f
            Tensor x = Tensor.create(
                    X_SHAPE,
                    FloatBuffer.wrap(ArrayHelper.flatten(dataSet))
            );



            long start = System.nanoTime();

            int times = 100;

            for(int t = 0; t < times; t++) {

                float[][] y = sess.runner()
                        .feed("X", x)
                        .fetch("y_")
                        .run()
                        .get(0)
                        .copyTo(new float[1][6]);


            }

            long end = System.nanoTime();

            System.out.println((end - start)/1e6 + " ms");

        }

    }
}