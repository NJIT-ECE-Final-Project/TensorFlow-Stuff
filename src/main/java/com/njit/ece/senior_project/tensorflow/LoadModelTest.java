package com.njit.ece.senior_project.tensorflow;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.FloatBuffer;
import java.util.List;

/**
 * Created by hp- on 8/5/2017.
 */
public class LoadModelTest {


    public static void main(String[] args) throws IOException {
        // good idea to print the version number, 1.2.0 as of this writing
        System.out.println(TensorFlow.version());

        final long[] X_SHAPE = new long[]{10, 128,9};

        float[] x_vals = new float[(int)X_SHAPE[0]*(int)X_SHAPE[1]*(int)X_SHAPE[2]];

        for(int i = 0; i < x_vals.length; i++) {
            x_vals[i] = 1.0f;
        }


        File trainedModelFile = new File(LoadModelTest.class.getResource("56").getFile());


        // load the model Bundle
        try (SavedModelBundle b = SavedModelBundle.load(trainedModelFile.getAbsolutePath(), "serve")) {

            // create the session from the Bundle
            Session sess = b.session();
            // create an input Tensor, value = 2.0f
            Tensor x = Tensor.create(
                    X_SHAPE,
                    FloatBuffer.wrap(x_vals)
            );


            // run the model and get the result, 4.0f.
            //Session.Runner runner = sess.runner();
            //runner = runner.feed("X", x);
            //runner = runner.fetch("y_",0);
            //List<Tensor> tensorList = runner.run();

            long start = System.nanoTime();

            int times = 1;

            for(int t = 0; t < times; t++) {

                float[][] y = sess.runner()
                        .feed("X", x)
                        .fetch("y_")
                        .run()
                        .get(0)
                        .copyTo(new float[10][6]);

            }

            long end = System.nanoTime();

            System.out.println((end - start)/1e6 + " ms");

        }
    }
}