package org.anticml;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;

import org.anticml.event.PerceptronEvent;
import org.anticml.PerceptronListener;

public class Perceptron {
	private double[] weights;
	private double learningRate;
	private List<PerceptronListener> listeners;
	private long elapsedTime;
	private int epoch;
	private int maxEpochs;
	private double targetError;
	private List<Double> listOfMse;
	
	/**
	 * Use threshold activation function: if x >= 0.5 Then output = 1 Else output = 0
	 */
	public static final int THRESHOLD = 0;
	/**
	 * Use Sigmoid activation function
	 */
	public static final int SIGMOID = 1;
	/**
	 * Use Hyperbolic tangent activation function
	 */
	public static final int HYPERBOLIC_TANGENT = 2;
	
	private int activationFunction = SIGMOID;
	
	public Perceptron(int numInputs) {
		weights = new double[numInputs];
		for(int i=0; i<numInputs; i++) {
			weights[i] = Math.random();
		}
		learningRate = 0.5;
		maxEpochs = 1000;
		targetError = 0.0001;
		listeners = new ArrayList<>();
		listOfMse = new ArrayList<>();
	}
	
	public void setLearningRate(double learningRate) { this.learningRate = learningRate; }
	public double getLearningRate() { return learningRate; }
	public double[] getWeights() { return weights; }
	public void setWeights(double[] weights) { this.weights = weights; }
	public void setActivationFunction(int activationFunction) { this.activationFunction = activationFunction; }
	public int getActivationFunction() { return activationFunction; }
	public int getEpochs() { return epoch; }
	public void setMaxEpochs(int maxEpochs) {
		this.maxEpochs = maxEpochs;
	}
	public int getMaxEpochs() { return maxEpochs; }
	public void setTargetError(double targetError) { this.targetError = targetError; }
	public double getTargetError() { return targetError; }
	public List getListOfMse() { return listOfMse; }
	
	public double calculate(double[] input) {
		if( input.length != weights.length)
			throw new RuntimeException("Number of inputs does not match the number of weights.");
		// Multiply the weights by the input
		double[] mul = multiply(weights, input);
		// Add them up
		double weightSum = sum(mul);
		// Apply the activation function
		double activation = activationFunction(weightSum);
		
		// Return the output
		return activation;
	}
	
	public void train(double[][] inputs, double[] desiredOutput) {
		if( inputs[0].length != weights.length )
			throw new RuntimeException("Number of columns in inputs does not match the number of input weights.");
		PerceptronEvent e = new PerceptronEvent(this, 
				weights,
				0,
				0,
				0,
				0);
		fireTrainingStarted(e);
		long startTime = System.currentTimeMillis();
		elapsedTime = 0;
		
		double[] outputs = new double[inputs.length];
		double error = 0;
		int epoch = 0;
		double mseError = 0;
		listOfMse.clear();
		
		do {
			epoch++;
			// For each training example
			for(int i=0; i<inputs.length; i++) {
				// Feed in a training sample
				outputs[i] = calculate(inputs[i]);
				// Calculate error
				error = desiredOutput[i] - outputs[i];
				// Update mean squared error sum
				mseError += error;
				// Update the weights. This weight update rule is from the Wikipedia
				// article about the Perceptron
				// 
				for(int j=0; j<weights.length; j++) {
					weights[j] = weights[j] + learningRate * error * inputs[i][j];
				}
				// Notify listeners of an epoch
				e = new PerceptronEvent(this, 
						weights,
						outputs[i],
						desiredOutput[i],
						calcMSE(mseError, inputs.length),
						(double)elapsedTime/1000d);
				fireTrainingInProgress(e);

				// calculate elapsed time
				elapsedTime = System.currentTimeMillis() - startTime;
			}
			
			listOfMse.add(calcMSE(mseError, inputs.length));
			// If the minimum error value has been reached then exit loop
			if( mseError <= targetError || epoch >= maxEpochs)
				break;
			mseError = 0;

		} while( true );
		
		// Fire training complete event
		e = new PerceptronEvent(this, 
				weights,
				0,
				0,
				0,
				(double)elapsedTime/1000d);
		fireTrainingComplete(e);
	}
	private double calcMSE(double errorSum, int numInputs) {
		return Math.pow(errorSum/numInputs, 2);
	}
	private double[] multiply(double[] a, double[] b) {
		double[] result = new double[a.length];
		for(int i=0; i<a.length; i++) {
			result[i] = a[i] * b[i];
		}
		return result;
	}
	
	private double sum(double[] a) {
		double sum = 0;
		for(int i=0; i<a.length; i++) {
			sum += a[i];
		}
		return sum;
	}
	
	private double[] add(double[] a, double b) {
		double[] result = new double[a.length];
		for(int i=0; i<a.length; i++) {
			result[i] = a[i] + b;
		}
		return result;
	}
	
	private double activationFunction(double weightSum) {
		switch(activationFunction) {
			case SIGMOID:
				return 1 / (1 + Math.exp(-weightSum));
			case HYPERBOLIC_TANGENT:
				return ( Math.exp(2*weightSum) - 1) / ( Math.exp(2*weightSum) + 1);
			case THRESHOLD:
				if( weightSum >= 0.5 )
					return 1;
				else return 0;
			}
		return 0;
	}
	public double getElapsedTimeInSeconds() {
		return (double)elapsedTime/ 1000d;
	}
	
	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append("Weights: ");
		sb.append(Arrays.toString(weights));
		return sb.toString();
	}
	
	public void addListener(PerceptronListener perceptronListener) {
		listeners.add(perceptronListener);
	}
	public void removeListener(PerceptronListener perceptronListener) {
		listeners.remove(perceptronListener);
	}
	protected void fireTrainingStarted(PerceptronEvent perceptronEvent) {
		for(int i=0; i<listeners.size(); i++)
			listeners.get(i).trainingStarted(perceptronEvent);
	}
	protected void fireTrainingInProgress(PerceptronEvent perceptronEvent) {
		for(int i=0; i<listeners.size(); i++)
			listeners.get(i).trainingInProgress(perceptronEvent);
	}
	protected void fireTrainingComplete(PerceptronEvent perceptronEvent) {
		for(int i=0; i<listeners.size(); i++)
			listeners.get(i).trainingComplete(perceptronEvent);
	}
	
	public static void main(String[] args) {
		// Create a Perceptron with two inputs
		Perceptron p = new Perceptron(2); 
		
		// Register a listener to report training progress
		p.addListener(new PerceptronListener() {
			// Called when training has started
			public void trainingStarted(PerceptronEvent e) { 
				Perceptron p = e.getPerceptron();
				System.out.println("---------------Initialise Perceptron---------------");
				System.out.println("Weights: " + Arrays.toString(p.getWeights()));
			}
			// Repeatedly called whilst training is in progress. Report some useful stats
			public void trainingInProgress(PerceptronEvent e) {
				Perceptron p = e.getPerceptron();
				System.out.println();
				
				double[] weights = e.getWeights();
				for(int i=0; i<weights.length; i++)
					System.out.print(weights[i] + " ");
				
				System.out.print(" OUTPUT: "+e.getOutput());
				System.out.print(" REAL OUTPUT: " + e.getRealOutput());
				System.out.println(" ERROR: " + e.getMeanSquaredError());
				
				
			}
			// Called when training is complete
			public void trainingComplete(PerceptronEvent e) {
				Perceptron p = e.getPerceptron();
				System.out.println();
				System.out.println("------------------- MEAN SQUARED ERRORS BY EPOCH-------------");
				List l = p.getListOfMse();
				for(int i=0; i<l.size(); i++)
					System.out.println(l.get(i));
				System.out.println("Training completed in " + e.getElapsedTimeInSeconds() + " seconds.");
			}
		});
		
		// Learn AND function
		double[][] trainingSamples = {{0,0},{1,0},{0,1},{1,1}};
		double[] desiredOutput = {0,1,1,1};
		
		// Set max epochs before training stops. This prevents the perceptron going into
		// an infinite loop in the case where it is given a non linear classification problem
		p.setMaxEpochs(25);
		// Set the minimum target error that will stop training
		p.setTargetError(0.001);
		// Learning rate
		p.setLearningRate(0.5);
		p.setActivationFunction(Perceptron.SIGMOID);
		p.train(trainingSamples, desiredOutput);
		
	}
}
