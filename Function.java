package org.anticml.math;
/**
 * An interface that should be implemented for custom function mappings. The matrix
 * class has a <code>map(Function f)</code> method that will apply a function to
 * each element of the matrix and return the resulting matrix.
 *  
 * @author Mario
 *
 */
public interface Function {

	/**
	 * Calculate some function and return the result. The return value
	 * is used by the Matrix class to map each element of the matrix
	 * with the function.
	 * 
	 * @return The result
	 * @see Matrix#map(Function)
	 */
	public double calculate(double x);
}
