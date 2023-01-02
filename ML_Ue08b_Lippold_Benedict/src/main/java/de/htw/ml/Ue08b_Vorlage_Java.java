package de.htw.ml;

import java.io.IOException;

import de.htw.ml.data.CreditDataset;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.layout.HBox;
import javafx.stage.Stage;

/**
 * This class does not have any TODOs.
 * It trains and tests an expert system.
 * 
 * @author Nico Hezel
 */
public class Ue08b_Vorlage_Java {
	
	private static final int TrainingIterations = 3000;
	private static final float LearnRate = 0.5f;
	
	public static void main(String[] args) throws IOException {
		
		// read the data
		final CreditDataset dataset = new CreditDataset();
		final int[] categories = dataset.getCategories();
		
		// expert system
		ExpertSystem expert = new ExpertSystem(TrainingIterations, LearnRate, categories);
		
		// train the individual logistic regressions
		expert.train(dataset);
		
		// let the whole system make a prediction for the test data
		double predictionRate = expert.test(dataset);
		System.out.printf("Overall prediction rate %.2f %n", predictionRate);
		
		// plot the prediction rates and train errors
		FXApplication.plot(expert.getPredictionRates(), expert.getTrainErrors(), categories);
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
	
		private static float[][] predictionRatesPerLabel;
		private static float[][] trainingsErrorPerLabel;
		private static int[] labels;
		
		/**
		 * Start the application and plot the data
		 * 
		 * @param predictionRates
		 * @param trainingsError
		 * @param uniqueValues
		 */
		public static void plot(float[][] predictionRates, float[][] trainingsError, int[] uniqueValues) {
			predictionRatesPerLabel = predictionRates;
			trainingsErrorPerLabel = trainingsError;
			labels = uniqueValues;
		}
		
		/**
		 * Draw the plot
		 */	
		@Override public void start(Stage stage) {
	
			HBox pane = new HBox(10, getPredictionRateChart(), getTrainingsErrorChart());		
			Scene scene = new Scene(pane, 1000, 400);
			
			stage.setTitle("Chart");		
			stage.setScene(scene);
			stage.show();
	    }
		
		@SuppressWarnings("unchecked")
		protected LineChart<Number, Number> getTrainingsErrorChart() {
			
			final NumberAxis xAxis = new NumberAxis();
			xAxis.setLabel("iteration");
	        final NumberAxis yAxis = new NumberAxis();
	        yAxis.setLabel("trainings error");
	        
			final LineChart<Number, Number> sc = new LineChart<>(xAxis, yAxis);
			sc.setAnimated(false);
			sc.setCreateSymbols(false);
			
			for (int labelIndex = 0; labelIndex < trainingsErrorPerLabel.length; labelIndex++) {
				float[] predictionRates = trainingsErrorPerLabel[labelIndex];
				if(predictionRates == null) continue;
	
				XYChart.Series<Number, Number> series = new XYChart.Series<>();
				series.setName("Label "+labels[labelIndex]);
				for (int i = 0; i < predictionRates.length; i++) 
					series.getData().add(new XYChart.Data<Number, Number>(i, predictionRates[i]));			
				sc.getData().addAll(series);
			}	
			return sc;
		}
		
		@SuppressWarnings("unchecked")
		protected LineChart<Number, Number> getPredictionRateChart() {
			
			final NumberAxis xAxis = new NumberAxis();
			xAxis.setLabel("iteration");
	        final NumberAxis yAxis = new NumberAxis(0, 100, 10);
	        yAxis.setLabel("prediction rate");
	        
			final LineChart<Number, Number> sc = new LineChart<>(xAxis, yAxis);
			sc.setAnimated(false);
			sc.setCreateSymbols(false);
			
			for (int labelIndex = 0; labelIndex < predictionRatesPerLabel.length; labelIndex++) {
				float[] predictionRates = predictionRatesPerLabel[labelIndex];
				if(predictionRates == null) continue;
				
				XYChart.Series<Number, Number> series = new XYChart.Series<>();
				series.setName("Label "+labels[labelIndex]);
				for (int i = 0; i < predictionRates.length; i++) 
					series.getData().add(new XYChart.Data<Number, Number>(i, predictionRates[i]));			
				sc.getData().addAll(series);
			}	
			return sc;
		}
	}
}
