package org.anticml;

import java.util.ArrayList;

import org.anticml.event.LinearRegressionEvent;
import org.anticml.math.Matrix;
import org.anticml.math.NonInvertibleMatrixException;

/**
 * An implementation of linear regression that can handle multiple variables. There is
 * another, simpler linear regression class in this package called <code>SimpleLinearRegression</code>
 * that should be used if your data set has only one variable because it is more efficient.
 * 
 * @author Mario Gianota 3 July 2019 gianotamario@gmail.com
 *
 */
public class MultivariateLinearRegression {
	/*
	 * The learning rate to use in the case we are using gradient descent
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

    /**
     * Construct a new SimpleLinearRegression object.
     */
    public MultivariateLinearRegression() {
        exitStrategy = ExitStrategy.CONVERGENCE;
    }


    /**
     * Trains the model and calculates the y-intercept and slope.
     * <code>features</code> is a <code>Matrix</code> object containing the
     * features, <code>labels</code> is a <code>Matrix</code> object containing the labeled
     * target data, <code>theta</code> 
     *
     * @param features The sample data set
     * @param labels The label target data
     * @param theta The y-intercept and slope
     * @param learningRate The learning rate
     * @see #setExitStrategy(ExitStrategy)
     * @see #setMaxIterations(int)
     * @see #setUseGradientDescent(boolean)
     * @exception NonInvertibleMatrixException If the data set cannot be inverted
     */
    public void train(Matrix features, Matrix labels, Matrix theta,
                      double learningRate) {
        // sanity check parameters
        if( labels.getColumns() != 1)
            throw new IllegalArgumentException("Labels must have only one column.");
        if( features.getRows() != labels.getRows() )
            throw new IllegalArgumentException("Features and Labels must have equal number of rows.");
        if( theta.getRows() != features.getColumns())
            throw new IllegalArgumentException("Number of rows in theta must equal the number of columns in features.");
        if( learningRate <= 0)
            throw new IllegalArgumentException("Learning rate must be greater than 0.");

        double[][] d = features.getMatrixArray();
        boolean allOnes = true;
        for(int row=0; row<features.getRows(); row++) {
            if( d[row][0] != 1.0d) {
                allOnes = false;
                break;
            }
        }

        if( allOnes == false)
            throw new IllegalArgumentException("Column zero in feature set must be set to 1.");

        x = features;
        y = labels;
        this.theta = theta;
        this.learningRate = learningRate;
        double cost = 0;

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

    private void gradientDescent() {
        double m = x.getRows();
        
        Matrix temp = x.dot(theta);
        Matrix error = temp.subtract(y);
        Matrix newX = error.transpose().dot(x);
        theta = theta.subtract(newX.transpose().multiply(learningRate/m));
        
    }

    private double computeCost() {
        double m = theta.getRows();
        return ( 1 / (2*m)) * x.dot(theta).subtract(y).square().sum();
    }

    /**
     * Make a prediction given the supplied input. Call this method after the train method
     * has been called. 
     * @param input The datapoint input
     * @return the prediction
     */
    public double predict(Matrix input) {

        Matrix prediction = input.dot(theta);
        return prediction.getValueAt(0, 0);
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
