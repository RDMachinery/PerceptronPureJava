package org.anticml.event;

import org.anticml.Perceptron;

public class PerceptronEvent {
	Perceptron source;
	double[] weights;
	double output;
	double realOutput;
	double error;
	double elapsedTime;
	double mseError;
	
	public PerceptronEvent(Perceptron source, 
			double[] weights,
			double output,
			double realOutput,
			double mseError, 
			double elapsedTime) {
		this.source = source;
		this.mseError = mseError;
		this.elapsedTime = elapsedTime;
		this.weights = weights;
		this.output = output;
		this.realOutput = realOutput;
	}
	public double[] getWeights() {return weights; }
	public double getOutput() { return output; }
	public double getRealOutput() { return realOutput; }
	public Perceptron getPerceptron() { return source; }
	public double getMeanSquaredError() { return mseError; }
	public double getElapsedTimeInSeconds() { return elapsedTime; }
}
