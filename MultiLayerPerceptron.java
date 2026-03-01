package org.anticml;
import java.util.Arrays;

import org.anticml.math.Matrix;
/**
 * An implementation of a feed forward multi-layer perceptron with one input layer, one
 * hidden layer and an output layer. The number of input nodes, hidden nodes and output
 * nodes can all be configured.
 * 
 * @author Mario
 *
 */
public class MultiLayerPerceptron {

	/*
	 * Number of input nodes.
	 */
	private int inputNodes;
	/*
	 * Number of hidden nodes.
	 */
	private int hiddenNodes;
	/*
	 * Number of output nodes
	 */
	private int outputNodes;

	/*
	 * A matrix representing the weights from the input layer to the hidden layer
	 */
	private Matrix weightsIH;
	/*
	 * A matrix representing the weights from the hidden layer to the output layer 
	 */
	private Matrix weightsHO;
	/*
	 * The biases for the hidden laer
	 */
	private Matrix biasH;
	/*
	 * The biases for the output layer
	 */
	private Matrix biasO;
	/*
	 * The hidden outputs of the network. Populated after each call to feedFoward(inputs).
	 */
	private Matrix hiddenOutputs;
	/*
	 * The learning rate
	 */
	private double learningRate;
	/*
	 * The number of epochs, updated by the train() method.
	 */
	private int epochs;
	
	/*
	 * No empty constructor allowed
	 */
	private MultiLayerPerceptron() {}
	
	/**
	 * Creates a new multi-layer perceptron with the given number of
	 * input nodes, hidden nodes and output nodes.
	 * 
	 * @param numInputs The number of input nodes
	 * @param numHidden The number of hidden nodes
	 * @param numOutputs The number of output nodes
	 */
	public MultiLayerPerceptron(int numInputs, int numHidden, int numOutputs) {
		inputNodes = numInputs;
		hiddenNodes = numHidden;
		outputNodes = numOutputs;
	
		weightsIH = new Matrix(hiddenNodes, inputNodes);
		weightsHO = new Matrix(outputNodes, hiddenNodes);
		weightsIH.randomize();
		weightsHO.randomize();
	
		biasH = new Matrix(hiddenNodes, 1);
		biasO = new Matrix(outputNodes, 1);
		biasH.randomize();
		biasO.randomize();
	
		learningRate = 0.1;
		epochs = 1;
	}
	public int getNumInputNodes() { return inputNodes; }
	public int getNumHiddenNodes() { return hiddenNodes; }
	public int getNumOutputNodes() { return outputNodes; }
	public Matrix getInputToHiddenWeights() { return weightsIH; }
	public Matrix getHiddenBiases() { return biasH; }
	public Matrix getHiddenToOutputWeights() { return weightsHO; }
	public Matrix getOutputBiases() { return biasO; }
	public void setInputToHiddenWeights(Matrix weights) { weightsIH = weights; }
	public void setHiddenBiases(Matrix biases) { biasH = biases; }
	public void setHiddenToOutputWeights(Matrix weights) { weightsHO = weights; }
	public void setOutputBiases(Matrix biases) { biasO = biases; }
	
	/**
	 * Set the number of times the network should iterate through the training set
	 * during training.
	 * 
	 * @param epochs The number of epochs to execute
	 */
	public void setEpochs(int epochs) { this.epochs = epochs; }
	/**
	 * Returns the number of epochs the training method executed.
	 * @return Number of epochs
	 */
	public int getEpochs() { return epochs; }
	
	/**
	 * Feeds input data into the network and returns the output. The length of <code>x</code>
	 * must match the number of input nodes. The length of the returned output is equal to
	 * the number of output nodes.
	 * 
	 * @param x The input data
	 * @return The network's output
	 */
	public double[] feedForward(double[] x) {
		
		Matrix inputs = Matrix.fromArray(x);
		// Hidden output = Whidden * Inputs + Bias
		Matrix hidden = weightsIH.dot(inputs).add(biasH);
		hidden = sigmoid(hidden);
		
		hiddenOutputs = hidden.copy(); // Save hidden outputs for use by the backprop
		
		Matrix output = weightsHO.dot(hidden).add(biasO);
		output = sigmoid(output);
		
		return output.toArray();
	}
	/**
	 * Returns the number of weights between the input and hidden layer.
	 * 
	 * @return number of weights
	 */
	public int getNumInputToHiddenWeights() {
		return this.hiddenNodes * this.inputNodes;
	}
	/**
	 * Returns the number of weights between the hidden and output layer.
	 * 
	 * @return number of weights
	 */
	public int getNumHiddenToOutputWeights() {
		return this.outputNodes * this.hiddenNodes;
	}
	/**
	 * Returns the number of hidden bias weights.
	 * 
	 * @return number of weights
	 */
	public int getNumHiddenBiasWeights() {
		return biasH.getRows();
	}
	
	/**
	 * Returns the number of output bias weights
	 * 
	 * @return number of weights
	 */
	public int getNumOutputBiasWeights() {
		return biasO.getRows();
	}
	
	public void train(double[][] inputs, double[] target) {
		if( inputs.length != target.length )
			throw new IllegalArgumentException("Number of rows in inputs must equal number of rows in target.");
		
		for(int i=0; i<epochs; i++) {
			for(int j=0; j<inputs.length; j++) {
				double[] t = new double[1];
				t[0] = target[j];
				backprop(inputs[j], t);
			}
		}
	}
	
	private void backprop(double[] input, double[] target) {
		// Get the output from the network & convert to a Matrix
		Matrix outputs = Matrix.fromArray(feedForward(input));
		Matrix targets = Matrix.fromArray(target);
		
		// Calculate the output errors. error = targets - output
		Matrix outputErrors = targets.subtract(outputs);
		
		
		// Calculate weight deltas
		Matrix gradient = dsigmoid(outputs);
		// Calculate output gradient
		gradient = gradient.multiply(outputErrors);
		gradient = gradient.multiply(learningRate);
		// Calculate hidden to output deltas
		Matrix weightHODeltas = gradient.dot(hiddenOutputs.transpose());
		
		// Update hidden to output weights + bias
		weightsHO = weightsHO.add(weightHODeltas);
		biasO = biasO.add(gradient);
		
		// Calculate the hidden errors - note the transpose
		Matrix hiddenErrors = weightsHO.transpose().dot(outputErrors);
		Matrix hiddenGradient = dsigmoid(hiddenOutputs);
		// Calculate hidden gradient
		hiddenGradient = hiddenGradient.multiply(hiddenErrors);
		hiddenGradient = hiddenGradient.multiply(learningRate);
		// Calculate input to hidden deltas
		Matrix inputT = Matrix.fromArray(input).transpose();
		Matrix weightIHDeltas = hiddenGradient.dot(inputT);
		
		// Update input to hidden weights + bias
		weightsIH = weightsIH.add(weightIHDeltas);
		biasH = biasH.add(hiddenGradient);
	}
	public static final Matrix dsigmoid(Matrix m) {
		double[][] data = m.getMatrixArray();
		for(int i=0; i<data.length; i++) {
			for(int j=0; j<data[0].length; j++) {
				data[i][j] = data[i][j] * ( 1 - data[i][j]);
			}
		}
		return m;
	}
	private Matrix sigmoid(Matrix m) {
		double[][] a = m.getMatrixArray();
		for(int i=0; i<a.length; i++) {
			for(int j=0; j<a[0].length; j++) {
				a[i][j] = 1 / (1 + Math.exp(-a[i][j]));
			}
		}
		return m;
	}
	
	public static void main(String[] args) {
		MultiLayerPerceptron mlp = new MultiLayerPerceptron(2,6,1);
		double[][] inputs = {{0,0},{0,1},{1,0},{1,1}};
		double[] targets = {0,1,1,0};
		mlp.setEpochs(20000);
		mlp.train(inputs, targets);
	
		double[][] testSet = {{0,0},
				{1,0},
				{0,1},
				{1,1}
		};
		double[] predictions = new double[testSet.length];
		double[] score = new double[testSet.length];
		
		for( int i=0; i<testSet.length; i++ ) {
			predictions[i] = mlp.feedForward(testSet[i])[0];
			int j = 0;
			if( predictions[i] >= 0.5)
				j = 1;
			if( j == targets[i])
				score[i] = 1;
			System.out.println(predictions[i]); 
		}
		System.out.println("Score card: "+Arrays.toString(score));
		// Calculate accuracy total correct predictions / total number of predictions
		double tp = 0;
		for(int i=0; i<score.length; i++) {
			if( score[i] == 1)
				tp++;
		}
		double accuracy = tp / (double)predictions.length;
		
		System.out.println("Accuracy: "+accuracy);
	}
	
}
