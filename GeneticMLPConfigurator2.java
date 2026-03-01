package org.anticml;

	import java.util.ArrayList;

	import java.util.Arrays;
	import java.util.List;
	import java.util.Collections;
	import java.util.Random;
	
	/**
	 * An implementation of a genetic algorithm used to configure the weights
	 * and biases of a MultiLayerPerceptron. This method of initialising the
	 * weights and biases of a neural network is *highly* experimental, but the
	 * research literature indicates that this method is a perfectly acceptable
	 * alternative to backpropagation that is known to produce good results.
	 * <p>
	 * To use it, construct an instance of this class and supply the MLP network
	 * you wish to configure along with the inputs and targets. The inputs and targets
	 * are those arrays that you would normally feed into the MLP during training. They are used by 
	 * the configurator to asses the fitness of the generated solutions. See the constructor for more.
	 * 
	 * @author Mario
	 *
	 */
	public class GeneticMLPConfigurator2 {

		private int numGenerations;
		private List<Genome> genomes;
		MultiLayerPerceptron neuralNet;
		private double[][] inputs;
		private double[] targets;
		private double mutationRate;
		private int maxPopulationSize; // Maximum size the population can grow to
		private Genome fittestSolution;
		
		/**
		 * Constructs a new configurator that will configure the neural network supplied in the
		 * constructor. The <code>inputs</code> are the data one would normally feed into a
		 * network to generate a prediction, whilst the targets are the actual 'answers' to
		 * the inputs. The configurator uses the inputs and targets to asses the fitness of
		 * generated genomes (solutions). <code>maxPopulationSize</code> is the maximum number
		 * of genomes you wish the genetic algorithm to generate to find the fittest solution.
		 * 
		 * @param nn The neural net to configure
		 * @param inputs The inputs to be fed into the network
		 * @param targets The targets to compare against the inputs to asses fitness
		 * @param maxPopulationSize The maximum number of genomes the configurator will create
		 */
		public GeneticMLPConfigurator2(MultiLayerPerceptron nn,
				double[][] inputs,
				double[] targets,
				int maxPopulationSize) {
			if( maxPopulationSize % 2 != 0)
				throw new IllegalArgumentException("Initial population size must be an even number.");
			if( maxPopulationSize < 4 )
				throw new IllegalArgumentException("Population too small. Initial population size must be greater than 4.");
			neuralNet = nn;
			this.inputs = inputs;
			this.targets = targets;
			this.maxPopulationSize = maxPopulationSize;
			
			// Calculate genome size
			int genomeSize = nn.getNumInputToHiddenWeights() +
					nn.getNumHiddenBiasWeights() +
					nn.getNumHiddenToOutputWeights() +
					nn.getNumOutputBiasWeights();
			
			// If the size of the genome is not even then crossover won't find a midpoint
			if( genomeSize % 2 != 0)
				genomeSize += 1;
			
			// Create population
			genomes = new ArrayList<>();
			for(int i=0; i<maxPopulationSize; i++) {
				Genome g = new Genome(genomeSize);
				genomes.add(g);
			}
			
			mutationRate = 0.01;
		}
		/**
		 * Sets the rate of mutation. The rate of mutation is a probability. The probability
		 * is used to compute the chance that each individual "gene" in each of the genomes is
		 * mutated.<p>
		 * 
		 * The mutation rate really effects how the algorithm behaves because it is responsible
		 * for the amount of variation of the information in the genetic pool. If there is no mutation then
		 * there is a very high probability that the algorithm will become stuck in a local
		 * minimum. Too much mutation will result in the genomes being hopelessly randomised
		 * ensuring that the GA will never converge on a solution. The default rate of mutation is
		 * set to 0.01 which means a 1% chance of each gene being mutated. The general
		 * rule of thumb for altering this number is to alter in tiny increments and observe the effect
		 * on the fitness of generated genomes.
		 * 
		 * @param rate The rate of mutation
		 */
		public void setMutationRate(double rate) { mutationRate = rate; }
		
		/**
		 * Returns the mutation rate.
		 * 
		 * @return rate of mutation
		 */
		public double getMutationRate() { return mutationRate; }
		
		/**
		 * Runs the configurator and returns the configured network. <code>maxGenerations</code>
		 * can be set to limit the number of generations that the genetic algorithm loops through.
		 * When <code>maxenerations</code> have been generated this method will return.
		 * 
		 * @param maxGenerations Maximum number of generations to loop through before returning
		 * @return The configured MLP network
		 */
		public MultiLayerPerceptron configure(int maxGenerations) {
			
			do {
				// Selection - Calculate individual fitness for population
				for(int j=0; j<genomes.size(); j++) 
					genomes.get(j).calcFitness(neuralNet, inputs, targets);
				Collections.sort(genomes);
				System.out.println("Generation #"+numGenerations);
				//for(int i=0; i<4; i++) {
					System.out.println("Fitness: "+genomes.get(0).getFitness());
				//}
				if( numGenerations >= maxGenerations )
					break;

				// Create mating pool
				List<Genome> matingPool = createMatingPool();
				// Crossover
				List<Genome> children = crossover(matingPool);
				
				// Mutate
				mutate(children);
			
				// New population from children
				this.populateNextGeneration(children);
			
				numGenerations++;
			} while( true );
			Collections.sort(genomes);
			fittestSolution = genomes.get(0);
			System.out.println("Exit loop. Fittest = "+fittestSolution);
			// Set the weights and biases on the neural network and return it
			
			neuralNet.setInputToHiddenWeights(fittestSolution.getInputToHiddenWeights(neuralNet));
			neuralNet.setHiddenBiases(fittestSolution.getHiddenBiases(neuralNet));
			neuralNet.setHiddenToOutputWeights(fittestSolution.getHiddenToOutputWeights(neuralNet));
			neuralNet.setOutputBiases(fittestSolution.getOutputBiases(neuralNet));
			
			return neuralNet;
		}
		public Genome getFittestSolution() { return fittestSolution; }
		
		private List<Genome> createMatingPool() {
			List<Genome> matingPool = new ArrayList<>();
			
			// This code implements the Tournament selection method of selecting partners
			// Select n individuals at random from the population and then select the best
			// of these to become a parent
			List<Genome> randomGroup = new ArrayList<Genome>();
			
			Random rnd = new Random();
			int index = 0;
			int n = (int) (0.5 * genomes.size());
			for(int i=0; i<n; i++) {
				for(int j=0; j<n; j++) {
					index = rnd.nextInt(genomes.size());
					randomGroup.add(genomes.get(index));
				}
				Genome fittest = getFittest(randomGroup);
				matingPool.add(fittest);
				randomGroup.clear();
			}
			return matingPool;
		
		}
		
		private Genome getFittest(List<Genome> group) {
			double max = 0;
			Genome g = null;
			for(int i=0; i<group.size(); i++) {
				if( group.get(i).getFitness() >= max) {
					max = group.get(i).getFitness();
					g = group.get(i);
				}
			}
			return g;
		}
		
		private List<Genome> crossover(List<Genome> matingPool) {
			List<Genome> children = new ArrayList<>();
			int index = 0;
			
			do {
				Genome partnerA = matingPool.get(index);
				Genome partnerB = matingPool.get(index+1);
				Genome[] g = partnerA.crossover(partnerB);
				children.add(g[0]);
				//children.add(g[1]);
				index+=1;
				if( index >= matingPool.size() - 1)
					break;
			} while( true );
			return children;
		}

		private void mutate(List<Genome> children) {
			for(int i=0; i<children.size(); i++) {
				children.get(i).mutate(mutationRate);
			}
		}

		private void populateNextGeneration(List<Genome> children) {
			// The size of the next population is equal to
			// maxPopulationSize/numChildren * numChildren
			//List<Genome> newPopulation = new ArrayList<>();
			genomes.clear();
			
			int newPopulationSize = (maxPopulationSize/children.size()) * children.size();
			int blockSize = maxPopulationSize/children.size();
			for(int i=0; i <= blockSize; i++) {
				for(int j=0; j<children.size(); j++) {
					genomes.add( children.get(j).copy());
				}
			}
			
			int i = 0;
			// Add extra children if needed
			while( genomes.size() <= maxPopulationSize) {
				genomes.add(children.get(i).copy());
				i++;
				if( i >= children.size() )
					i = 0;
			}
			
		}
}

