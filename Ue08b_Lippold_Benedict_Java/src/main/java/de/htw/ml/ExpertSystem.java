package de.htw.ml;

import java.util.Arrays;

import org.jblas.FloatMatrix;

import de.htw.ml.data.CreditDataset;
import de.htw.ml.data.Dataset;

/**
 * There are some TODO here. 
 * Combines the different logistic regression modules into one system.
 * 
 * @author Nico Hezel
 */
public class ExpertSystem {

	protected LogisticRegression regression;
	
	protected int[] categories; 			// the system should be able to predict these labels
	protected FloatMatrix[] thetas; 		// weights of a logistic regression model for a label
	protected float[][] predictionRates;	// the prediction rates during training
	protected float[][] trainErrors;		// the error rates during training

	
	public ExpertSystem(int trainingIterations, float learnRate, int[] categories) {
		this.regression = new LogisticRegression(trainingIterations, learnRate);
		this.thetas = new FloatMatrix[categories.length];
		this.predictionRates = new float[categories.length][];
		this.trainErrors = new float[categories.length][];		
		this.categories = categories;
	}
	
	/**
	 * Trains for every unique label a separate logistic regression model.
	 * 
	 * @param dataset
	 */
	public void train(CreditDataset dataset) {
		
		// train a logistic regression for each category
		for (int i = 0; i < categories.length; i++) {	
			final int category = categories[i];
			
			// create the training set for this category
			final Dataset subset = dataset.getSubset(category);
			final float ratio = (subset.getYTrain().sum() / subset.getYTrain().rows * 100);
			System.out.printf("Train category %d (%.2f%% share with %d elements)\n", category, ratio, subset.getYTrain().getRows());
			
			// start the training process
			thetas[i] = regression.train(subset.getXTest(), subset.getYTest(), subset.getXTrain(), subset.getYTrain());
			predictionRates[i] = regression.getPredictionRates();
			trainErrors[i] = regression.getTrainError();
			System.out.printf("Best prediction rate %.2f%%\n\n", (new FloatMatrix(predictionRates[i])).max());
		}
	}
		
	/**
	 * TODO Missing the search for the strongest prediction
	 * 
	 * @param dataset
	 * @return
	 */
	public float test(Dataset dataset) {
		
		FloatMatrix xTest = dataset.getXTest();
		FloatMatrix yTest = dataset.getYTest();
		
		// the predictions for each label
		FloatMatrix[] hypothesisArr = Arrays.stream(thetas).map(theta -> LogisticRegression.predict(xTest, theta)).toArray(FloatMatrix[]::new);
		
		// run through all predictions ...
		int correctSum = 0;
		for (int r = 0; r < yTest.getRows(); r++) {
			int expectedLabel = (int)yTest.data[r];
			
			// TODO ... and find the strongest one (highest value) 
			float hypothesisLabel = -1;

			// count how many times the system found the right label
			if(expectedLabel == hypothesisLabel)
				correctSum++;
		}
		return (float)correctSum / yTest.getRows();
	}
	
	public float[][] getPredictionRates() {
		return predictionRates;
	}
	
	public float[][] getTrainErrors() {
		return trainErrors;
	}
}
