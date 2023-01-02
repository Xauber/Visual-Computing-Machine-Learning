package de.htw.ml;

import java.io.IOException;
import org.jblas.FloatMatrix;
import org.jblas.util.Random;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;

public class Ue07_Lippold_Benedict {

    // TODO change the names of the axis
    public static final String title = "Credit approximation";
    public static final String xAxisLabel = "Iteration count";
    public static final String yAxisLabel = "Prediction percentage";

    public static void main(String[] args) throws IOException {
        approximateCreditsGivenSigmoid();
    }

    private static void approximateCreditsGiven() throws IOException {
        FloatMatrix germanCredit = FloatMatrix.loadCSVFile("german_credit_jblas.csv");

        // max Values for normalization
        float[] germanCreditMaxValues = new float[germanCredit.columns];

        for(int i = 0; i < germanCreditMaxValues.length; i++) {
            germanCreditMaxValues[i] = germanCredit.getColumn(i).max();
        }

        // normalization
        FloatMatrix[] germanCreditNormalizedVector = new FloatMatrix[germanCredit.columns];

        for(int i = 0; i < germanCreditNormalizedVector.length; i++) {
            germanCreditNormalizedVector[i] = new FloatMatrix(germanCredit.getColumn(i).div(germanCreditMaxValues[i]).toArray());
        }

        // matrix from germanCredit
        FloatMatrix xValueMatrix = germanCreditNormalizedVector[1];
        for(int i = 2; i < germanCreditNormalizedVector.length; i++) {
            xValueMatrix = FloatMatrix.concatHorizontally(xValueMatrix, germanCreditNormalizedVector[i]);
        }

        // alpha values & table length
        float[] learningRates = {0.005f, 0.01f, 0.05f, 0.1f};
        int dataTableLength = germanCredit.rows;

        // RMSE array for different alphas
        float[][] rmseValues = new float[4][100];
        float [][] creditsGivenPercentage = new float[4][100];
        float[] creditsGivenHypothesisValues = null;


        // different curves for learning rates
        for(int i = 0; i < learningRates.length; i++) {
            Random.seed(5);

            // initialize thetas
            float[] thetaValues = new float[germanCredit.columns - 1];
            for(int j = 0; j < thetaValues.length; j++) {
                thetaValues[j] = Random.nextFloat();
            }

            FloatMatrix thetaValuesVector = new FloatMatrix(thetaValues);

            // initialize smallest rmse
            float smallestRmse = 100000.0f;

            // iterations to find best thetas
            for(int j = 0; j < 100; j++) {
                // Calculate approximated mpg values with weighted function (hypothesis function)
                FloatMatrix creditAmountHypothesis = xValueMatrix.mmul(thetaValuesVector);
                creditsGivenHypothesisValues = creditAmountHypothesis.toArray();

                // Disparity
                float[] disparity = new float[dataTableLength];
                for(int k = 0; k < dataTableLength; k++) {
                    disparity[k] = creditsGivenHypothesisValues[k] - germanCredit.getColumn(0).toArray()[k];
                }

                // disparity vector
                FloatMatrix disparityVector = new FloatMatrix(disparity);

                // theta deltas & normalize theta deltas
                FloatMatrix transposedValueMat = xValueMatrix.transpose();
                FloatMatrix thetaDeltaVector = transposedValueMat.mmul(disparityVector);
                FloatMatrix normalizedThetaDeltaVector = thetaDeltaVector.mul(learningRates[i] / dataTableLength);

                // set new theta values & update thetas
                FloatMatrix newThetaValuesVector = thetaValuesVector.sub(normalizedThetaDeltaVector);
                thetaValuesVector = newThetaValuesVector;

                //  calculate RMSE
                float squareErrorSum = 0;

                for(int k = 0; k < dataTableLength; k++) {
                    squareErrorSum += Math.pow(creditsGivenHypothesisValues[k] - germanCredit.getColumn(0).toArray()[k], 2);
                }
                float mse = squareErrorSum / dataTableLength;
                float rmse = (float) Math.sqrt(mse);

                // store current rmse & update best thetas if rmse is smaller
                rmseValues[i][j] = rmse;

                if(rmse < smallestRmse) {
                    smallestRmse = rmse;
                }

                // Binarize
                float creditsGiven = 0;
                for(int k = 0; k < creditsGivenHypothesisValues.length; k++) {
                    if (creditsGivenHypothesisValues[k] >= 0.5) {
                        creditsGivenHypothesisValues[k] = 1;
                        creditsGiven++;
                    }
                    else{
                        creditsGivenHypothesisValues[k] = 0;
                    }
                }
                creditsGivenPercentage[i][j] = creditsGiven / germanCredit.rows * 100.0f;
            }
            System.out.println("RMSE for creditsGiven learning curve " + i +": " + smallestRmse);
        }
        // Plot
        FXApplication.plot(creditsGivenPercentage[3]);
        Application.launch(FXApplication.class);
    }

    private static void approximateCreditsGivenSigmoid() throws IOException {
        FloatMatrix germanCredit = FloatMatrix.loadCSVFile("german_credit_jblas.csv");

        // max Values for normalization
        float[] germanCreditMaxValues = new float[germanCredit.columns];

        for(int i = 0; i < germanCreditMaxValues.length; i++) {
            germanCreditMaxValues[i] = germanCredit.getColumn(i).max();
        }

        // normalization
        FloatMatrix[] germanCreditNormalizedVector = new FloatMatrix[germanCredit.columns];
        for(int i = 0; i < germanCreditNormalizedVector.length; i++) {
            germanCreditNormalizedVector[i] = new FloatMatrix(germanCredit.getColumn(i).div(germanCreditMaxValues[i]).toArray());
        }

        // matrix from germanCredit
        FloatMatrix xValueMatrix = germanCreditNormalizedVector[1];
        for(int i = 2; i < germanCreditNormalizedVector.length; i++) {
            xValueMatrix = FloatMatrix.concatHorizontally(xValueMatrix, germanCreditNormalizedVector[i]);
        }

        float testSetSize = 0.1f;
        FloatMatrix germanCreditsTest = xValueMatrix.getRange(0, (int) (xValueMatrix.rows * testSetSize), 0, xValueMatrix.columns);
        FloatMatrix germanCreditsTraining = xValueMatrix.getRange((int) (xValueMatrix.rows * testSetSize), xValueMatrix.rows, 0, xValueMatrix.columns);

        // alphas & tables length
        float[] learningRates = {0.1f, 0.5f, 1.0f, 2.0f, 4.0f};
        int datenwertlänge = germanCreditsTraining.rows;
        int iterationAmount = 500;

        // RMSE array for different alphas
        float[][] rmseValues = new float[learningRates.length][iterationAmount];
        float [][] creditsGivenPercentage = new float[learningRates.length][iterationAmount];

        // different curves for learning rates
        for(int i = 0; i < learningRates.length; i++) {
            Random.seed(5);

            float[] thetaValues = new float[xValueMatrix.columns];
            int thetaMin = -1;
            int thetaMax = 1;
            for(int j = 0; j < thetaValues.length; j++) {
                thetaValues[j] = thetaMin + Random.nextFloat() * (thetaMax - thetaMin);
            }

            FloatMatrix thetaValuesVector = new FloatMatrix(thetaValues);

            // intitialize smallest rmse
            float smallestRmse = 100000.0f;

            // iterations to find best thetas
            for(int j = 0; j < iterationAmount; j++) {

                // Hypothesis
                FloatMatrix creditsGivenHypothesisTraining = germanCreditsTraining.mmul(thetaValuesVector);
                FloatMatrix creditsGivenHypothesisTest = germanCreditsTest.mmul(thetaValuesVector);
                float[] creditsGivenHypothesisValuesTraining = creditsGivenHypothesisTraining.toArray();
                float[] creditsGivenHypothesisValuesTest = creditsGivenHypothesisTest.toArray();

                // sqeeuze Hypothesis into Sigmoid function
                float[] creditsGivenHypothesisValuesSigmoidTraining = new float[creditsGivenHypothesisTraining.length];
                float[] creditsGivenHypothesisValuesSigmoidTest = new float[creditsGivenHypothesisTest.length];
                for(int h = 0; h < creditsGivenHypothesisValuesTraining.length; h++) {
                    creditsGivenHypothesisValuesSigmoidTraining[h] = (float) (1 / (1 + Math.pow(Math.E, -creditsGivenHypothesisValuesTraining[h])));
                }
                for(int h = 0; h < creditsGivenHypothesisValuesTest.length; h++) {
                    creditsGivenHypothesisValuesSigmoidTest[h] = (float) (1 / (1 + Math.pow(Math.E, -creditsGivenHypothesisValuesTest[h])));
                }

                // Disparity
                float[] disparity = new float[datenwertlänge];
                for(int k = 0; k < datenwertlänge; k++) {
                    disparity[k] = creditsGivenHypothesisValuesSigmoidTraining[k] - germanCredit.getColumn(0).toArray()[k];
                }

                // create disparity vector & theta delta vector
                FloatMatrix disparityVector = new FloatMatrix(disparity);
                FloatMatrix transposedxValueMatrix = germanCreditsTraining.transpose();
                FloatMatrix thetaDeltaVector = transposedxValueMatrix.mmul(disparityVector);

                // normalize thetas & set new thetas
                FloatMatrix normalizedThetaDeltaVector = thetaDeltaVector.mul(learningRates[i] / datenwertlänge);
                thetaValuesVector = thetaValuesVector.sub(normalizedThetaDeltaVector);

                // calculate RMSE
                float squareErrorSum = 0;
                for(int k = 0; k < datenwertlänge; k++) {
                    squareErrorSum += Math.pow(creditsGivenHypothesisValuesTraining[k] - germanCredit.getColumn(0).toArray()[k], 2);
                }

                float mse = squareErrorSum / datenwertlänge;
                float rmse = (float) Math.sqrt(mse);

                // store current rmse & update best thetas if rmse is smaller
                rmseValues[i][j] = rmse;
                if(rmse < smallestRmse) {
                    smallestRmse = rmse;
                }

                //binarize and calculate the prediction error
                float predError = 0;
                for(int k = 0; k < creditsGivenHypothesisValuesSigmoidTest.length; k++) {
                    if (creditsGivenHypothesisValuesSigmoidTest[k] >= 0.5) {
                        creditsGivenHypothesisValuesSigmoidTest[k] = 1;
                    }
                    else{
                        creditsGivenHypothesisValuesSigmoidTest[k] = 0;
                    }
                    predError = predError +  Math.abs(creditsGivenHypothesisValuesSigmoidTest[k] - germanCredit.getColumn(0).toArray()[k]);
                }

                creditsGivenPercentage[i][j] = (germanCreditsTest.rows - predError) / germanCreditsTest.rows * 100;
            }
            System.out.println("RMSE for creditsGiven learning curve " + i +": " + smallestRmse);
        }
        // Plot
        FXApplication.plot(creditsGivenPercentage[2]);
        Application.launch(FXApplication.class);
    }


    // ---------------------------------------------------------------------------------
    // ------------ Alle Änderungen ab hier geschehen auf eigene Gefahr ----------------
    // ---------------------------------------------------------------------------------

    /**
     * We need a separate class in order to trick Java 11 to start our JavaFX application without any module-path settings.
     * https://stackoverflow.com/questions/52144931/how-to-add-javafx-runtime-to-eclipse-in-java-11/55300492#55300492
     *
     * @author Nico Hezel
     *
     */
    public static class FXApplication extends Application {

        /**
         * equivalent to linspace in Octave
         *
         * @param lower
         * @param upper
         * @param num
         * @return
         */
        private static FloatMatrix linspace(float lower, float upper, int num) {
            float[] data = new float[num];
            float step = Math.abs(lower-upper) / (num-1);
            for (int i = 0; i < num; i++)
                data[i] = lower + (step * i);
            data[0] = lower;
            data[data.length-1] = upper;
            return new FloatMatrix(data);
        }

        // y-axis values of the plot
        private static float[] dataY;

        /**
         * Draw the values and start the UI
         */
        public static void plot(float[] yValues) {
            dataY = yValues;
        }

        /**
         * Draw the UI
         */
        @SuppressWarnings("unchecked")
        @Override
        public void start(Stage stage) {

            stage.setTitle(title);

            final NumberAxis xAxis = new NumberAxis();
            xAxis.setLabel(xAxisLabel);
            final NumberAxis yAxis = new NumberAxis();
            yAxis.setLabel(yAxisLabel);

            final LineChart<Number, Number> sc = new LineChart<>(xAxis, yAxis);

            XYChart.Series<Number, Number> series1 = new XYChart.Series<>();
            series1.setName("Data");
            for (int i = 0; i < dataY.length; i++) {
                series1.getData().add(new XYChart.Data<Number, Number>(i, dataY[i]));
            }

            sc.setAnimated(false);
            sc.setCreateSymbols(true);

            sc.getData().addAll(series1);

            Scene scene = new Scene(sc, 500, 400);
            stage.setScene(scene);
            stage.show();
        }
    }
}
