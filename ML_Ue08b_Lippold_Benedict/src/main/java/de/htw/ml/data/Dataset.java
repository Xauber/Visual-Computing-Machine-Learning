package de.htw.ml.data;

import org.jblas.FloatMatrix;

/**
 * Dataset full train and test data
 * 
 * @author Hezel
 */
public interface Dataset {

	/**
	 * x-values of the train data
	 * @return
	 */
	public FloatMatrix getXTrain();

	/**
	 * y-values of the train data
	 * @return
	 */
	public FloatMatrix getYTrain();

	/**
	 * x-values of the test data
	 * @return
	 */
	public FloatMatrix getXTest();

	/**
	 * y-values of the test data
	 * @return
	 */
	public FloatMatrix getYTest();
}
