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
		System.out.println("Training model");
		this.predictionRates = new float[trainingIterations];
		this.trainErrors = new float[trainingIterations];
		
		// Initialize the weights
		org.jblas.util.Random.seed(7);
		FloatMatrix theta = FloatMatrix.rand(xTrain.getColumns(), 1);

		// Best combination of weights
		FloatMatrix bestTheta = theta.dup();
		float bestPredictionRate = 0.0f;

		float learningRate = 0.1f;

		// Training for x iterations
		for (int iteration = 0; iteration < trainingIterations; iteration++) {
			FloatMatrix prediction = predict(xTrain, theta);

			// Calculating the disparity
			float[] disparity = new float[xTrain.rows];

			for(int k = 0; k < xTrain.rows; k++) {
				disparity[k] = prediction.data[k] - yTrain.data[k];
			}

			// Create disparity vector & theta delta vector
			FloatMatrix disparityVector = new FloatMatrix(disparity);
			FloatMatrix transposedXTrain = xTrain.transpose();
			FloatMatrix thetaDelta = transposedXTrain.mmul(disparityVector);

			// Normalize thetas & set new thetas
			FloatMatrix normalizedThetaDeltaVector = thetaDelta.mul(learningRate / xTrain.rows);
			theta = theta.sub(normalizedThetaDeltaVector);

			float cost = cost(prediction, yTrain);

			predictionRates[iteration] = predictionRate(prediction, yTrain);
			trainErrors[iteration] = cost;
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
		// Calculate approximated y values with weighted function (hypothesis function)
		FloatMatrix predict = x.mmul(theta);
		predict = sigmoidi(predict);

		return predict;
	}
		
	/**
	 * Calculates the training error according to the logistical cost function or RMSE.
	 * 
	 * @param prediction
	 * @param y
	 * @return
	 */
	public static float cost(FloatMatrix prediction, FloatMatrix y) {
		// RMSE
		float squareErrorSum = 0;
		for(int k = 0; k < prediction.rows; k++) {
			squareErrorSum += Math.pow(prediction.data[k] - y.data[k], 2);
		}
		float mse = squareErrorSum / prediction.rows;
		float rmse = (float) Math.sqrt(mse);
		return rmse;
	}

	/**
	 * Calculates a prediction rate between the prediction and the desired result Y.
	 * 
	 * @param prediction
	 * @param y
	 * @return
	 */
	public static float predictionRate(FloatMatrix prediction, FloatMatrix y) {
		int correctPredictions = 0;
		float predictionRate;

		// Berechnung der Anzahl richtiger Vorhersagen, wenn binarisiert
		for(int pred = 0; pred < prediction.rows; pred++) {
			if(Math.floor(prediction.data[pred] + 0.5f) == y.data[pred]) {
				correctPredictions++;
			}
		}

		// Berechnung des Prozentsatzes der richtigen Vorhersagen
		predictionRate = correctPredictions / (float) prediction.rows * 100;
		return predictionRate;
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