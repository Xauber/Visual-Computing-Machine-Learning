package de.htw.ml.data;

import java.io.IOException;
import java.util.*;
import java.util.stream.IntStream;

import org.jblas.FloatMatrix;
import org.jblas.JavaBlas;

/**
 * There are a lot TODOs here.
 * The class divides the german credit dataset into train and test data.
 *
 * @author Nico Hezel
 */
public class CreditDataset implements Dataset {
    protected Random rnd = new Random(7);

    protected FloatMatrix xTrain;
    protected FloatMatrix yTrain;

    protected FloatMatrix xTest;
    protected FloatMatrix yTest;

    protected int[] categories;

    public CreditDataset() throws IOException {
        System.out.println("Creating dataset");
        int predictColumn = 15; // type of apartment
        FloatMatrix data = FloatMatrix.loadCSVFile("german_credit_jblas.csv");

        // Liste mit allen Kategorien zur Prediction
        final FloatMatrix outputData = data.getColumn(predictColumn);
        categories = IntStream.range(0, outputData.rows).map(idx -> (int) outputData.data[idx]).distinct().sorted().toArray();
        int[] categorySizes = IntStream.of(categories).map(v -> (int) outputData.eq(v).sum()).toArray();
        System.out.println("The unique values of y are " + Arrays.toString(categories) + " and there number of occurrences are " + Arrays.toString(categorySizes));

        // Array with all rows that are not predictColumn
        int[] xColumns = IntStream.range(0, data.columns).filter(value -> value != predictColumn).toArray();

        // Input and output data
        FloatMatrix x = data.getColumns(xColumns);
        FloatMatrix y = data.getColumn(predictColumn);

        // min, maximum für alle spalten
        FloatMatrix xMin = x.columnMins();
        FloatMatrix xMax = x.columnMaxs();

        // normalize the data sets and add the bias column
        FloatMatrix xNorm = x.subRowVector(xMin).diviRowVector(xMax.sub(xMin));
        xNorm = FloatMatrix.concatHorizontally(FloatMatrix.ones(xNorm.rows, 1), xNorm);

        // Erstellen eines Trainings- und eines Testsatzes mit 90% bzw. 10% aller Daten
        // Testsatz enthält Bilder aus allen Kategorien in gleicher Menge
        int testDataPerCategory = data.getRows() / 10 / categories.length; // 10% test set
        int testDataCount = testDataPerCategory * categories.length;
        System.out.println("Use " + testDataCount + " as test data with " + testDataPerCategory + " elements per category.\n");
        int[] testRowIndices = new int[testDataCount];
        int testsFound = 0;

        // Indizes für die erforderliche Anzahl von Zeilen speichern, die die Kategorien enthalten
        for (int kategorie = 1; kategorie < categories.length + 1; kategorie++) {
            int datenInKategorie = 0;
            boolean kategorieFertig = false;
            for (int i = 0; i < y.rows && !kategorieFertig; i++) {
                if (y.data[i] == kategorie) {
                    datenInKategorie++;
                    testRowIndices[testsFound] = i;
                    testsFound++;
                    // Suche nach der aktuellen Kategorie beenden, wenn genügend Sätze gefunden wurden
                    if (datenInKategorie == testDataPerCategory) {
                        kategorieFertig = true;
                    }
                }
            }
        }


        // Hinzufügen aller Zeilenindizes zu den Trainingsdaten, die nicht im Array der Testindizes enthalten sind
        ArrayList<Integer> trainRowIndicesList = new ArrayList<>();
        int[] temp = IntStream.range(0, xNorm.rows).toArray();
        for(int i: temp) {
            trainRowIndicesList.add(i);
        }
        for(int i = 0; i < testRowIndices.length; i++) {
            int indexToRemove = trainRowIndicesList.indexOf(testRowIndices[i]);
            trainRowIndicesList.remove(indexToRemove);
        }
        int[] trainRowIndices = new int[trainRowIndicesList.size()];
        for(int i = 0; i < trainRowIndicesList.size(); i++) {
            trainRowIndices[i] = trainRowIndicesList.get(i);
        }

        // TODO
        // Training set
        xTrain = xNorm.getRows(trainRowIndices);
        yTrain = y.getRows(trainRowIndices);
        // Test set
        xTest = xNorm.getRows(testRowIndices);
        yTest = y.getRows(testRowIndices);
    }

    public int[] getCategories() {
        return categories;
    }

    /**
     * The train data should contain as many train entries as possible but the ratio
     * between data points of the desired category and data points of a different category
     * should be 50:50. All Y data are binarized:
     *  - desired category = 1
     *  - other category = 0
     *
     * @param category
     * @return {x Matrix,y Matrix}
     */
    public Dataset getSubset(int category) {
        // Finding all the indices of the lines in which the desired category occurs.
        // Search as many other lines with a different category. Remove indices if
        // necessary, to ensure the size of both sets are the same

        // Retrieve all points from training data that match the required category
        int[] catRowIndices = IntStream.range(0, yTrain.rows).filter(value -> category == yTrain.data[value]).toArray();

        // Retrieve as many points from training data that don't match the category as those that match it
        // Limiting the array to maximally hold as many values as those that are not in the current category
        int numLeftoverIndices = yTrain.rows - catRowIndices.length;
        int[] catRowIndicesLimited = catRowIndices;

        if(catRowIndices.length > numLeftoverIndices) {
            System.out.println("Limiting non category array");
            int[] temp = new int[numLeftoverIndices];

            for(int i = 0; i < numLeftoverIndices; i++) {
                temp[i] = catRowIndices[i];
            }

            catRowIndicesLimited = temp;
        }

        // Mischen Sie die übrig gebliebenen Punkte, um eine Zufallsverteilung zu erhalten.
        ArrayList<Integer> leftoverIndicesList = new ArrayList<>();
        int[] temp = IntStream.range(0, yTrain.rows).toArray();

        for(int i: temp) {
            leftoverIndicesList.add(i);
        }

        // Shuffle list
        Collections.shuffle(leftoverIndicesList);

        // Alle Elemente, die zur Kategorie gehören, entfernen
        for(int i = 0; i < catRowIndices.length; i++) {
            int indexToRemove = leftoverIndicesList.indexOf(catRowIndices[i]);
            leftoverIndicesList.remove(indexToRemove);
        }

        // Fügen Sie so viele Indizes zur Liste der Kategorien hinzu, wie es Punkte in der Kategorieliste gibt.
        int[] notCatRowIndices = new int[catRowIndicesLimited.length];

        for(int i = 0; i < notCatRowIndices.length; i++) {
            notCatRowIndices[i] = leftoverIndicesList.get(i);
        }

        // Kombinieren von Datenpunkten der gewünschten Kategorie mit denen einer anderen Kategorie
        int[] rowIndices = new int[catRowIndicesLimited.length * 2];
        System.arraycopy(catRowIndicesLimited, 0, rowIndices, 0, catRowIndicesLimited.length);
        System.arraycopy(notCatRowIndices, 0, rowIndices, catRowIndicesLimited.length, notCatRowIndices.length);

        // Get the desired data points and binarize the Y-values
        return new Dataset() {
            @Override
            public FloatMatrix getXTrain() {
                return xTrain.getRows(rowIndices);
            }

            @Override
            public FloatMatrix getYTrain() {
                return yTrain.getRows(rowIndices).eq(category);
            }

            @Override
            public FloatMatrix getXTest() {
                return xTest;
            }

            @Override
            public FloatMatrix getYTest() {
                return yTest.eq(category);
            }
        };
    }

    @Override
    public FloatMatrix getXTrain() {
        return xTrain;
    }

    @Override
    public FloatMatrix getYTrain() {
        return yTrain;
    }

    @Override
    public FloatMatrix getXTest() {
        return xTest;
    }

    @Override
    public FloatMatrix getYTest() {
        return yTest;
    }
}
