package de.htw.ml.data;

import org.junit.BeforeClass;
import org.junit.Test;

import de.htw.ml.data.CreditDataset;
import de.htw.ml.data.Dataset;

import org.jblas.FloatMatrix;
import org.junit.Assert;

/**
 * Test the dataset class and its functions
 * 
 * @author Nico Hezel
 */
public class CreditDatasetTest {

	public static CreditDataset dataset;
	
	@BeforeClass
	public static void setUp() throws Exception {
		dataset = new CreditDataset();
	}
	
	@Test
	public void categoryCountTest() {
		Assert.assertEquals(3, dataset.getCategories().length);
	}
	
	@Test
	public void categoryTest() {
		int[] categories = dataset.getCategories();
		for (int i = 0; i < categories.length; i++) 			
			Assert.assertEquals(i+1, categories[i]);
	}
	
	@Test
	public void testsetCountTest() {
		Assert.assertEquals(99, dataset.getXTest().getRows());
		Assert.assertEquals(99, dataset.getYTest().getRows());
	}
	
	@Test
	public void testsetEqualLabelTest() {
		int[] categories = dataset.getCategories();
		FloatMatrix yTest = dataset.getYTest();
		for (int category : categories) 			
			Assert.assertEquals(33, (int)yTest.eq(category).sum());
	}
	
	@Test
	public void createTestsetCountTest() {
		int[] categories = dataset.getCategories();
		
		for (int category : categories) { 			
			Dataset subset = dataset.getSubset(category);
			Assert.assertEquals(99, subset.getXTest().getRows());
			Assert.assertEquals(99, subset.getYTest().getRows());
		}
	}
	
	@Test
	public void createTestsetBinaryTest() {
		int[] categories = dataset.getCategories();
		
		for (int category : categories) { 			
			Dataset subset = dataset.getSubset(category);
			FloatMatrix yTest = subset.getYTest();
			Assert.assertEquals(33, (int) yTest.sum());
		}
	}
	
	@Test
	public void trainsetCountTest() {
		Assert.assertEquals(901, dataset.getXTrain().getRows());
		Assert.assertEquals(901, dataset.getYTrain().getRows());
	}
	
	@Test
	public void createTrainingssetBinaryTest() {
		int[] categories = dataset.getCategories();
		
		for (int category : categories) { 			
			Dataset subset = dataset.getSubset(category);
			FloatMatrix yTrain = subset.getYTrain();			
			Assert.assertEquals(yTrain.getRows(), ((int) yTrain.sum()) * 2 );
		}
	}
}
