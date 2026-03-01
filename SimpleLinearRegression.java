package org.anticml;
import org.anticml.math.Matrix;
import org.anticml.math.NonInvertibleMatrixException;
import org.anticml.event.LinearRegressionEvent;
import java.util.ArrayList;

/**
 * This class provides an implementation of a simple (one feature) Linear
 * Regression model. By default it uses the Normal Equation method to find
 * the y-intercept and slope though this can be changed to use the gradient descent algorithm.
 * Just call <code>setUseGradientDescent(true)</code> and the class will switch to that
 * algorithm.
 * <p>
 * If your data has more than
 * one feature then you will need to use the <code>MultivariateLinearRegression</code>
 * class which can perform linear regression on input data containing more
 * than one feature.
 * <p>
 * Example usage:
 * <p>
 * <code>
 *       SimpleLinearRegression slr = new SimpleLinearRegression();
 *		 
 *       // Sample training data
 *       Matrix dataSet = new Matrix();
 *
 *       // Label target data
 *       Matrix labels = new Matrix();
 *       
 *       // Matrix of y-intercept and slope, initialised to 0.0d;
 *       Matrix theta = new Matrix(2,1);
 *
 *       
 *       double[][] xData = {{1,1},{1,2},{1,3}};
 *       dataSet.readData(xData);
 *       
 *       double[][] yData = {{1},{2},{3}};
 *       labels.readData(yData);
 *       
 *       
 *       // By default the class uses the Normal Equation so set the
 *       // learning rate to 0d because that only applies when using
 *       // gradient descent.
 *
 *        slr.train(dataSet, labels, theta, 0d);
 *
 *		// Make a prediction
 *		try {
 *			double prediction = slr.predict(1.0d);
 *			System.out.println("prediction = " + prediction);
 *		} catch(NonInvertibleMatrixException e) {
 *	    }
 *			
 *
 * </code>
 *
 * @author Mario Gianota 30 July 2019 gianotamario@gmail.com
 * @see Matrix
 * @see MultivariateLinearRegression
 */
public class SimpleLinearRegression {
	/*
	 * The leqarning rate to use in the case we are using gradient descent
	 */
    private double learningRate;
    /*
     * Maximum number of iterations
     */
    private int maxIters;
    /*
     * Main loop exit strategy
     */
    private ExitStrategy exitStrategy;
    /*
     * Number of iterations
     */
    private int numIters;
    /*
     * Listeners notified of events
     */
    private ArrayList<LinearRegressionListener> listeners;
    /*
     * The training data Matrix
     */
    private Matrix x;
    /*
     * The label target data matrix
     */
    private Matrix y;
    /*
     * y-intercept and slope
     */
    private Matrix theta;
    /*
     * The last cost saved
     */
    private double lastCost;
    /*
     * Flag to use gradient descent instead of the Normal Equation
     */
    private boolean useGradientDescent;

    /**
     * Construct a new SimpleLinearRegression object.
     */
    public SimpleLinearRegression() {
        exitStrategy = ExitStrategy.CONVERGENCE;
        useGradientDescent = false;
    }

    /**
     * Use gradient descent instead of the Normal Equation. By default, the Normal Equation
     * is used to calculate the y-intercept and slope however, if your training data is very large
     * say greater than 3,000 samples, or your data matrix is non invertible then it
     * you should use gradient descent. Set <code>useGradientDescent</code> to true
     * if this is the case.
     *
     * @param useGradientDescent Set true if you wish to use gradient descent
     */
    public void setUseGradientDescent(boolean useGradientDescent) {
        this.useGradientDescent = useGradientDescent;
    }

    /**
     * Trains the model and calculates the y-intercept and slope.
     * <code>dataset</code> is a <code>Matrix</code> object containing the
     * feature, <code>labels</code> is a <code>Matrix</code> object containing the labelled
     * target data, <code>theta</code> must be a 2x1 column vector matrix containing the initial
     * values for the y-intercept and the slope (usually both values are set to 0)
     * and <code>learningRate</code> controls how
     * big a step each iteration of the gradient descent will take. If the Normal Equation is being
     * used (which is the default) then this parameter is ignored and you should set it to some
     * zero value.
     * <code>dataSet</code> must be an nx2 dimension <code>Matrix</code> with all values in column zero 
     * set to 1 and <code>labels</code> must
     * be an nx1 dimension matrix. Both matrices must have an equal number of rows.
     * <code>theta</code> must be a 2x1 dimension matrix
     * <code>theta.getValueAt(0,0)</code> is assumed to
     * be the y-intercept and <code>theta.getValueAt(1,0)</code> is assumed to be the slope.
     * <p>
     *     When this method returns, <code>theta</code> will contain the calculated y-intercept
     *     and slope. 
     * </p>
     * <p>    
     *     In the case of using gradient descent note that the method uses one of two configurable 
     *     exit strategies to determine
     *     when the gradient descent loop should terminate and permit this method to return:
     *     <code>ExitStrategy.MAX_ITERATIONS</code> and <code>ExitStrategy.CONVERGENCE</code>. By
     *     default, the strategy is <code>ExitStrategy.CONVERGENCE</code> which means that the method returns when 
     *     the cost function converges to its minimum value. The exit
     *     strategy is only applicable if the gradient descent method is being used.
     * </p>
     *
     * @param dataSet The sample data set
     * @param labels The label target data
     * @param theta The y-intercept and slope
     * @param learningRate The learning rate
     * @see #setExitStrategy(ExitStrategy)
     * @see #setMaxIterations(int)
     * @see #setUseGradientDescent(boolean)
     * @exception NonInvertibleMatrixException If the data set cannot be inverted
     */
    public void train(Matrix dataSet, Matrix labels, Matrix theta,
                      double learningRate) throws NonInvertibleMatrixException {
        // sanity check parameters
        if( dataSet.getColumns() != 2)
            throw new IllegalArgumentException("Data set must have two columns.");
        if( labels.getColumns() != 1)
            throw new IllegalArgumentException("Labels must have only one column.");
        if( dataSet.getRows() != labels.getRows() )
            throw new IllegalArgumentException("Data set and Labels must have equal number of rows.");
        if( theta.getColumns() != 1)
            throw new IllegalArgumentException("Theta must have only one column.");
        if( learningRate <= 0)
            throw new IllegalArgumentException("Learning rate must be greater than 0.");

        double[][] d = dataSet.getMatrixArray();
        boolean allOnes = true;
        for(int row=0; row<dataSet.getRows(); row++) {
            if( d[row][0] != 1.0d) {
                allOnes = false;
                break;
            }
        }

        if( allOnes == false)
            throw new IllegalArgumentException("Column zero in data set must be set to 1.");

        x = dataSet;
        y = labels;
        this.theta = theta;
        this.learningRate = learningRate;
        double cost = 0;

        if( ! useGradientDescent ) {
            // Use Normal Equation
            applyNormalEquation();
            LinearRegressionEvent e = new LinearRegressionEvent(this, theta);
            fireEvent(e);
        } else {
            // Use gradient descent
            switch(exitStrategy) {
                case MAX_ITERATIONS:
                    while( numIters++ < maxIters ) {
                        cost = computeCost();
                        gradientDescent();
                        LinearRegressionEvent e = new LinearRegressionEvent(this,cost, theta);
                        fireEvent(e);
                    }
                    break;
                case CONVERGENCE:
                    while( (cost = computeCost()) != lastCost ) {
                        numIters++;
                        gradientDescent();
                        lastCost = cost;
                        LinearRegressionEvent e = new LinearRegressionEvent(this,cost, theta);
                        fireEvent(e);
                    }
                    break;
            }
        }
    }
    private void applyNormalEquation() throws NonInvertibleMatrixException {
        theta = x.dot(x.transpose().dot(x).inverse()).transpose().dot(y);
    }
    private void gradientDescent() {
        double m = x.getRows();

        double thetaZero = theta.getValueAt(0,0);
        double thetaOne = theta.getValueAt(1, 0);

        Matrix xx = x.getColumn(1);

        // h = theta(1) + (theta(2) * x)
        Matrix hypothesis = xx.multiply(thetaOne).add(thetaZero);

        double newThetaZero = thetaZero - learningRate * (1/m) * hypothesis.subtract(y).sum();
        double newThetaOne = thetaOne - learningRate * (1/m) * hypothesis.subtract(y).multiply(xx).sum();

        theta.setValueAt(0, 0, newThetaZero);
        theta.setValueAt(1, 0, newThetaOne);
    }

    private double computeCost() {
        double m = theta.getRows();
        Matrix predictions = x.dot(theta);

        // J = 1/(2*m) * sum((predictions - y) .^ 2);
        Matrix sqrErrors = predictions.subtract(y).square();
        double sum = sqrErrors.sum();
        return (1/(2*m)) * sum;

    }

    /**
     * Returns the slope. Call this method after the train method has been called to
     * obtain the calculated slope.
     * @return the slope
     */
    public double getSlope() { return theta.getValueAt(1,0); }

    /**
     * Returns the y-intercept. Call this method after the train method has been called to
     * obtain the calculated y-intercept.
     * @return the y-intercept
     */
    public double getYIntercept() { return theta.getValueAt(0,0);}

    /**
     * Make a prediction given the supplied input. Call this method after the train method
     * has been called. The prediction is calculated as <code>prediction = y-intercept + slope * input</code>
     * @param input The datapoint input
     * @return the prediction
     */
    public double predict(double input) {
        // prediction = theta0 + theta1 * input

        return theta.getValueAt(0, 0) + theta.getValueAt(1, 0) * input;
    }
    
    /**
     * Set the main loop exit strategy. The exit strategy determines how the main loop
     * in the train method will exit. Setting <code>ExitStrategy.MAX_ITERATIONS</code>
     * causes the loop to run for a set number of iterations before exiting. Setting
     * <code>ExitStrategy.CONVERGENCE</code> causes the main loop to exit when the
     * algorithm has reached convergence. This is the default exit strategy used.
     * 
     * @param es The exit strategy
     * @see #setMaxIterations(int)
     */
    public void setExitStrategy(ExitStrategy es) {
        exitStrategy = es;
    }
    
    /**
     * Set the maximum number of iterations the training loop will execute. This
     * setting will only take effect if <code>setExitStrategy</code> has been set
     * to <code>ExitStrategy.MAX_ITERATIONS, otherwise it is ignored.
     * 
     * @param maxIters maximum iterations
     * @see #setExitStrategy(ExitStrategy)
     */
    public void setMaxIterations(int maxIters) {
        this.maxIters = maxIters;
    }
    
    /**
     * Returns the exit strategy used by the main training loop to exit the loop.
     * 
     * @return exit strategy
     */
    public ExitStrategy getExitStrategy() {
        return exitStrategy;
    }
    
    /**
     * Returns the name of the algorithm being used to calculate the slope and
     * y-intercept. This class can use two algorithms: Gradient Descent and the
     * Normal Equation. By default, the Normal Equation is used.
     * 
     * @return algorithm name
     */
    public String getAlgorithmName() {
    	if( useGradientDescent )
    		return "Gradient Descent";
    	else
    		return "Normal Equation";
    }
    
    /**
     * Returns the maximum number of iterations that will be executed before the training loop will exit.
     * 
     * @return maximum iterations
     */
    public int getMaxIterations() {
        return maxIters;
    }
    
    /**
     * The number of elapsed iterations. The number of elapsed iterations is updated as
     * the training loop executes.
     * 
     * @return number of elapsed iterations
     */
    public int getNumIterations() {
        return numIters;
    }
    
    /**
     * Add a <code>LinearRegressionEvent</code> listener that will be notified as training is
     * in progress.
     * 
     * @param l The listener
     */
    public void addListener(LinearRegressionListener l) {
        if( listeners == null ) {
            listeners = new ArrayList<>();
        }
        listeners.add(l);
    }
    
    /**
     * Remove a <code>LinearRegressionEvent</code> listener. The listener will no longer be
     * notified of events.
     * 
     * @param l The listener
     */
    public void removeListener(LinearRegressionListener l) {
        if( listeners != null )
            listeners.remove(l);
    }

    /**
     * Notify listeners that an event has occurred.
     * 
     * @param e The event
     */
    protected void fireEvent(LinearRegressionEvent e) {
        if( listeners == null )
            return;
        for(int i=0; i< listeners.size(); i++) {
            LinearRegressionListener l = listeners.get(i);
            l.handleEvent(e);
        }
    }
}