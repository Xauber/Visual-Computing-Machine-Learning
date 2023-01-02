package de.htw.ml;

import org.jblas.FloatMatrix;

/**
 * A lot of TODOs here.
 * This is a simple logistic regression model.
 *  
 * @author Nico Hezel
 */
public class LogisticRegression {
	
	protected int trainingIterations;
	protected float learnRate;
	protected float[] predictionRates;
	protected float[] trainErrors;	
	
	public LogisticRegression(int trainingIterations, float learnRate) {
		this.trainingIterations = trainingIterations;
		this.learnRate = learnRate;
	}

	public FloatMatrix train(FloatMatrix xTest, FloatMatrix yTest, FloatMatrix xTrain, FloatMatrix yTrain) {
		this.predictionRates = new float[trainingIterations];
		this.trainErrors = new float[trainingIterations];
		
		// initialize the weights
		org.jblas.util.Random.seed(7);
		FloatMatrix theta = FloatMatrix.rand(xTrain.getColumns(), 1);
		
		// current training error
		trainErrors[0] = cost(predict(xTrain, theta), yTrain);

		// best combination of weights
		FloatMatrix bestTheta = theta.dup();
		float bestPredictionRate = predictionRates[0] = predictionRate(predict(xTest, theta), yTest);
		
		// training
		for (int iteration = 0; iteration < trainingIterations; iteration++) {
			
			// TODO training using the logistic regression	
			
			// TODO fill the prediction rate and train error arrays
			predictionRates[iteration] = 0;
			trainErrors[iteration] = 0;
		}
		
		return bestTheta;
	}

	/**
	 * Calculates a prediction of the input data X and the current weights theta
	 * 
	 * @param x
	 * @param theta
	 * @return
	 */
	public static FloatMatrix predict(FloatMatrix x, FloatMatrix theta) {
		// TODO Auto-generated method stub
		return null;
	}
		
	/**
	 * Calculates the training error according to the logistical cost function or RMSE.
	 * 
	 * @param prediction
	 * @param y
	 * @return
	 */
	public static float cost(FloatMatrix prediction, FloatMatrix y) {
		// TODO Auto-generated method stub
		return 0;
	}

	/**
	 * Calculates a prediction rate between the prediction and the desired result Y.
	 * 
	 * @param prediction
	 * @param y
	 * @return
	 */
	public static float predictionRate(FloatMatrix prediction, FloatMatrix y) {
		// TODO Auto-generated method stub
		return 0;
	}

	/**
	 * Prediction rates of the last training
	 * 
	 * @return
	 */
	public float[] getPredictionRates() {
		return predictionRates;
	}
	
	/**
	 * error rates of the last training
	 * 
	 * @return
	 */
	public float[] getTrainError() {
		return trainErrors;
	}
	
	/**
	 * Replaces the values in the Input Matrix with their sigmoid variant.
	 * 
	 * @param input
	 * @return
	 */
	public static FloatMatrix sigmoidi(FloatMatrix input) {
		for (int i = 0; i < input.data.length; i++)
			input.data[i] = (float) (1. / ( 1. + Math.exp(-input.data[i]) ));
		return input;
	}
}