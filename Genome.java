package org.anticml;
import java.util.Arrays;
import org.anticml.math.Matrix;

public class Genome implements Comparable<Genome>{
	double[] dna;
	double fitness;
	int genomeSize;
	
	public Genome(int genomeSize) {
		dna = new double[genomeSize];
		for(int j=0; j<genomeSize; j++) {
			dna[j] = Math.random() * 2 - 1;
		}
		this.genomeSize = genomeSize;
	}
	private Genome() {}
	
	public void calcFitness(MultiLayerPerceptron nn, double[][] inputs, double[] targets) { 
		// Set weights + biases
		Matrix weightsIH = getInputToHiddenWeights(nn);
		Matrix weightsHO = getHiddenToOutputWeights(nn);
		Matrix biasH = getHiddenBiases(nn);
		Matrix biasO = getOutputBiases(nn);
		nn.setInputToHiddenWeights(weightsIH);
		nn.setHiddenBiases(biasH);
		nn.setHiddenToOutputWeights(weightsHO);
		nn.setOutputBiases(biasO);

		// Feed inputs in and collect outputs and calculate output errors
		double[] outputs = new double[inputs.length];
		double[] errors = new double[outputs.length];


		for(int i=0; i<inputs.length; i++) {
			outputs[i] = nn.feedForward(inputs[i])[0];
			errors[i] = targets[i] - outputs[i];
		}
		
		Matrix mse = Matrix.fromArray(errors);
		fitness = mse.square().sum() * ( 1/(double)inputs.length);
		
	}
	public Genome copy() {
		Genome copy = new Genome();
		double[] cDNA = new double[dna.length];
		for(int i=0; i<dna.length; i++) 
			cDNA[i] = dna[i];
		copy.dna = cDNA;
		copy.fitness = fitness;
		return copy;
	}
	/**
	 * Crossover <code>partner</code> with this Genome and return a new child genome.
	 * 
	 * @param partner The genome partner
	 * @return Child genome
	 */
	public Genome[] crossover(Genome partner) {
		Genome child1 = new Genome(genomeSize);
		Genome child2 = new Genome(genomeSize);
		
		double[] aDna = dna;
		double[] bDna = partner.getDNA();
		
		double[] left = Arrays.copyOfRange(aDna, 0, aDna.length/2);
		double[] right = Arrays.copyOfRange(bDna, bDna.length/2, bDna.length);
		double[] cDna = join(left, right);
		child1.dna = cDna;
		
		left = Arrays.copyOfRange(bDna, 0, bDna.length/2);
		right = Arrays.copyOfRange(aDna, aDna.length/2, aDna.length);
		cDna = join(left, right);
		child2.dna = cDna;
		
		Genome[] g = new Genome[2];
		g[0] = child1;
		g[1] = child2;
		return g;
	}
	private static final double[] join(double[] a, double[] b) {
		double[] join = new double[a.length+b.length];
		int i = 0;
		for(i=0; i<a.length; i++)
			join[i] = a[i];
		for(int j=0; j<b.length; j++)
			join[j+i] = b[j];
		return join;
	}
	public void mutate(double mutationRate) {
		for(int i=0; i<dna.length; i++) {
			if( Math.random() < mutationRate ) {
				// Mutate the "gene"
				dna[i] = Math.random();
			}
		}
	}
	public Double getFitness() { return fitness; }
	public double[] getDNA() { return dna; }
	
	@Override
	public int compareTo(Genome genomeB) {
		return getFitness().compareTo(genomeB.getFitness());
	}

	public Matrix getInputToHiddenWeights(MultiLayerPerceptron nn) {
		int from = 0;
		int to = nn.getNumInputToHiddenWeights();
		double[] wData = Arrays.copyOfRange(dna, from, to);
		return unpackArrayToMatrix(wData);
	}
	public Matrix getHiddenBiases(MultiLayerPerceptron nn) {
		int from = nn.getNumInputToHiddenWeights() + 1;
		int to = from + nn.getNumHiddenBiasWeights();
		double[] bData = Arrays.copyOfRange(dna, from, to);
		return Matrix.fromArray(bData);
	}
	public Matrix getHiddenToOutputWeights(MultiLayerPerceptron nn) {
		int from = nn.getNumInputToHiddenWeights() +
				nn.getNumHiddenBiasWeights() + 1;
		int to = from + nn.getNumHiddenToOutputWeights();
		double[] wData = Arrays.copyOfRange(dna, from, to);
		return unpackArrayToMatrix(wData);
	}
	public Matrix getOutputBiases(MultiLayerPerceptron nn) {
		int from = nn.getNumInputToHiddenWeights() +
				nn.getNumHiddenBiasWeights() + 
				nn.getNumHiddenToOutputWeights() + 1;
		int to = from + nn.getNumOutputBiasWeights();
		double[] bData = Arrays.copyOfRange(dna, from, to);
		return Matrix.fromArray(bData);
	}
	
	private static final Matrix unpackArrayToMatrix(double[] array) {
		Matrix m = new Matrix();//System.out.println("Array length = " + array.length);
		int numColumns = 2;
		int numRows = array.length / numColumns;
		int row = 0;
		int column = 0;
		double[][] data = new double[numRows][numColumns];
		for(int i=0; i<array.length; i++ ) {
			//System.out.println("Row = " + row + " Column = "+column);
			data[row][column] = array[i];
			column++;
			if( column >= numColumns) {
				row++;
				column = 0;
			}
		}
		m.readData(data);
		return m;
	}
	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append("Fitness: " + fitness + " DNA: " + Arrays.toString(dna));
		return sb.toString();
	}
}
