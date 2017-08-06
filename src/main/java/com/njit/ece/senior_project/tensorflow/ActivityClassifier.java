package com.njit.ece.senior_project.tensorflow;

import com.njit.ece.senior_project.data.util.ArrayHelper;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.File;
import java.nio.FloatBuffer;

/**
 * Determines the activity being performed given the set of inertial signals
 */
public class ActivityClassifier {

    public enum Activity {
        WALKING(1),
        WALKING_UPSTAIRS(2),
        WALKING_DOWNSTAIRS(3),
        SITTING(4),
        STANDING(5),
        LAYING(6);

        private int num;

        Activity(int num) {
            this.num = num;
        }

        // numerical code for the Activity
        int getNum() {
            return num;
        }
    }

    // why can't Java just let you convert a number to a enum???
    private Activity activityFromInt(int n) {
        switch(n) {
            case 1:
                return Activity.WALKING;
            case 2:
                return Activity.WALKING_UPSTAIRS;
            case 3:
                return Activity.WALKING_DOWNSTAIRS;
            case 4:
                return Activity.SITTING;
            case 5:
                return Activity.STANDING;
            case 6:
                return Activity.LAYING;
            default:
                throw new IllegalArgumentException();
        }
    }

    private Session sess;


    public ActivityClassifier() {

        // hardcoded tensorflow model saved in the reosurces directory
        File trainedModelFile = new File(LoadModelTest.class.getResource("60").getFile());


        // create the session from the Bundle
        // load the model Bundle and run the data

        SavedModelBundle b = SavedModelBundle.load(trainedModelFile.getAbsolutePath(), "serve");

        // create the session from the Bundle
        sess = b.session();

    }


    public Activity classify(float[][][] signals) {

        long[] X_SHAPE = new long[]{signals.length, signals[0].length, signals[0][0].length};

        // create the input tensor
        Tensor x = Tensor.create(
                X_SHAPE,
                FloatBuffer.wrap(ArrayHelper.flatten(signals))
        );

        // get probablity of each classification
        float[][] probDistribution = sess.runner()
                .feed("X", x)
                .fetch("y_")
                .run()
                .get(0)
                .copyTo(new float[1][Activity.values().length]);

        // find the most likely classification from the probability distribution
        int maxIndex = 0;
        float maxVal = 0;
        for(int i = 0; i < probDistribution[0].length; i++) {
            if(probDistribution[0][i] > maxVal) {
                maxIndex = i;
                maxVal = probDistribution[0][maxIndex];
            }
        }

        return activityFromInt(maxIndex + 1 /*arrays start at 0...*/);

    }


    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        sess.close();
    }
}
