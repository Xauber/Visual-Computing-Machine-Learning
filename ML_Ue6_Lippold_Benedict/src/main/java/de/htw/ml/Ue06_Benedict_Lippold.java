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

public class Ue06_Benedict_Lippold {

    // TODO change the names of the axis
    public static final String title = "MPG Annäherung";
    public static final String xAxisLabel = "Iteration";
    public static final String yAxisLabel = "RMSE";

    public static void main(String[] args) throws IOException {
        FloatMatrix cars = FloatMatrix.loadCSVFile("cars_jblas.csv");


        // max Value speichern --> Achtung cars_jblas.csv" startet anders als das andere CSV file wieder direkt in Spalte 0 mit Zylinder-Werten!
        float cylindersMax = cars.getColumn(0).max();
        float displacementMax = cars.getColumn(1).max();
        float horsepowerMax = cars.getColumn(2).max();
        float weightMax = cars.getColumn(3).max();
        float accelerationMax = cars.getColumn(4).max();
        float modelYearMax = cars.getColumn(5).max();
        float mpgMax = cars.getColumn(6).max();

        // Normalisieren der Daten mit normMax Funktion
        float[] cylinders = cars.getColumn(0).div(cylindersMax).toArray();
        float[] displacement = cars.getColumn(1).div(displacementMax).toArray();
        float[] horsepower = cars.getColumn(2).div(horsepowerMax).toArray();
        float[] weight = cars.getColumn(3).div(weightMax).toArray();
        float[] acceleration = cars.getColumn(4).div(accelerationMax).toArray();
        float[] modelYear = cars.getColumn(5).div(modelYearMax).toArray();
        float[] mpg = cars.getColumn(6).div(mpgMax).toArray();

        // erzeugen der vektoren & der wertematrix
        FloatMatrix cylindersVector = new FloatMatrix(cylinders);
        FloatMatrix displacementVector = new FloatMatrix(displacement);
        FloatMatrix horsepowerVector = new FloatMatrix(horsepower);
        FloatMatrix weightVector = new FloatMatrix(weight);
        FloatMatrix accelerationVector = new FloatMatrix(acceleration);
        FloatMatrix modelYearVector = new FloatMatrix(modelYear);

        FloatMatrix spaltenwerteMatrix = new FloatMatrix(cylinders);
        spaltenwerteMatrix = FloatMatrix.concatHorizontally(spaltenwerteMatrix, displacementVector);
        spaltenwerteMatrix = FloatMatrix.concatHorizontally(spaltenwerteMatrix, horsepowerVector);
        spaltenwerteMatrix = FloatMatrix.concatHorizontally(spaltenwerteMatrix, weightVector);
        spaltenwerteMatrix = FloatMatrix.concatHorizontally(spaltenwerteMatrix, accelerationVector);
        spaltenwerteMatrix = FloatMatrix.concatHorizontally(spaltenwerteMatrix, modelYearVector);

        // gewünschte lernraten angeben
        float[] lernraten = {0.01f, 0.05f, 0.1f, 1.0f, 2.0f};

        int datenwertlänge = mpg.length;

        // RMSE array for each iteration
        float[][] rmseValues = new float[5][100];

        for(int i = 0; i < lernraten.length; i++) {

            Random.seed(7);

            // Theta Startwerte generieren --> für jede der 6 Spalten einen
            float[] thetaWerte = {Random.nextFloat(), Random.nextFloat(), Random.nextFloat(), Random.nextFloat(), Random.nextFloat(), Random.nextFloat()};

            FloatMatrix thetaWerteVec = new FloatMatrix(thetaWerte);

            float smallestRMSE = Float.POSITIVE_INFINITY;
            float[] bestThetas = new float[6];

            // schleife mit n iterationen um  möglichts gute Theta Werte zu finden, Schritte aus
            for(int j = 0; j < 100; j++) {

                // Hypothesis
                FloatMatrix mpgHypothesis = spaltenwerteMatrix.mmul(thetaWerteVec);
                float[] mpgHypothesisWerte = mpgHypothesis.toArray();

                // Disparity
                float[] disparity = new float[mpg.length];

                for(int k = 0; k < mpg.length; k++) {
                    disparity[k] = mpgHypothesisWerte[k] - mpg[k];
                }

                // Disparity Vektor
                FloatMatrix disparityVec = new FloatMatrix(disparity);

                
                // theta-delta werte
                FloatMatrix transpValueMatrix = spaltenwerteMatrix.transpose();
                FloatMatrix thetaDeltaVec = transpValueMatrix.mmul(disparityVec);
                FloatMatrix normThetaDeltaVec = thetaDeltaVec.mul(lernraten[i] / datenwertlänge);

                // Updaten der theta werte
                FloatMatrix newThetaWerteVec = thetaWerteVec.sub(normThetaDeltaVec);
                thetaWerteVec = newThetaWerteVec;

                // rmse
                float squareErrorSum = 0;
                for(int k = 0; k < mpg.length; k++) {
                    squareErrorSum += Math.pow(mpgMax * mpgHypothesisWerte[k] - mpgMax * mpg[k], 2);
                }
                float mse = squareErrorSum / mpg.length;
                float rmse = (float) Math.sqrt(mse);

                // speichern der rmse des jeweiligen durchlaufs
                rmseValues[i][j] = rmse;

                // kleinste rmse aktualisieren, wenn die aktuellen rmse kleiner sind & die besten Theta Werte aktualiseren
                if(rmse < smallestRMSE) {
                    smallestRMSE = rmse;
                    bestThetas = newThetaWerteVec.toArray();
                }
            }
            System.out.println("Beste RMSE nach letzem Durchlauf, mit Lernrate " + i + "  :  " + smallestRMSE);
            // FXApplication.plot(rmseValues[0]);
            // Application.launch(FXApplication.class);
        }

        FXApplication.plot(rmseValues[0]);
        FXApplication.plot(rmseValues[1]);
        FXApplication.plot(rmseValues[2]);
        // FXApplication.plot(rmseValues[3]);
        //FXApplication.plot(rmseValues[4]);
        Application.launch(FXApplication.class);

        // Leider hab ich nicht ganz gewusst wie ich mit der FXApplication mehrere Kurven plotten kann,
        // trotz plotten der RMSE werte der ersten 3 Lernraten wurde mir lediglich eine angezeigt.
        // Wenn ich die beiden weiteren lernraten 3 & 4 hinzugenommen habe, so wurde durch die unterschiedlichen Werte keine Kurve mehr angezeigt..
        // auch wenn ich immer innerhalb der aktuellen Iteration geplottet hatte, wurde mir leider keine Kurve mehr angezeigt


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
