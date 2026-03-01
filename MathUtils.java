package org.anticml.math;
import java.util.Random;

/**
 * A bunch of static methods that perform useful mathematical operations.
 * 
 * @author Mario Gianota gianotamario@gmail.com
 *
 */
public class MathUtils {
	final static Random rnd = new Random();
	final static double EPSILON = 1e-12;

	public MathUtils() { }
	/**
	 * Returns true if the given value lies within the given range.
	 * 
	 * @param value the value to test
	 * @param start the start range
	 * @param end the end range
	 * @return in range
	 */
	public boolean inRange(double value, double start, double end) {
		if( value >= start && value <= end)
			return true;
		return false;
	}

	/**
	 * Constrain a number between a lower bound and upper bound such that if <em>n</n>
	 * is less than <em>lower</em> then <em>lower</em> is returned or if <em>n</em> is
	 * greater than <em>upper</em> then <em>upper</em> is returned.
	 * 
	 * @param n the number to constrain.
	 * @param lower the lower bound
	 * @param upper the upper bound
	 * @return the number constrained
	 */
	public static double constrain(double n, double lower, double upper) {
		double num = n;
		if( n < lower )
			num = lower;
		else if( n > upper )
			num = upper;
		return num;
	}
	/**
	 * Constrain an integer number between a lower bound and upper bound such that if <em>n</n>
	 * is less than <em>lower</em> then <em>lower</em> is returned or if <em>n</em> is
	 * greater than <em>upper</em> then <em>upper</em> is returned.
	 * 
	 * @param n the number to constrain.
	 * @param lower the lower bound
	 * @param upper the upper bound
	 * @return the number constrained
	 */
	public static int constrain(int n, int lower, int upper) {
		int num = n;
		if( n < lower )
			num = lower;
		else if( n > upper )
			num = upper;
		return num;
	}
	/**
	 * Normalize a value to betwwen 0 and 1 for a given range.
	 */
	public static double normalize(double value, double min, double max) {
		return (value - min) / ( max - min);
	}
	/**
	 * Approach. Returns a value approaching the goal value based on current value
	 * and an offset.
	 * 
	 * @param goal the goal value
	 * @param current the current value
	 * @param deltaT Amount to add or subtract to current value to approach goal
	 */
	public static double approach(double goal, double current, double deltaT) {
		double difference = goal - current;
		if( difference > deltaT) {
			return current + deltaT;
		}
		if( difference < -deltaT) {
			return current - deltaT;
		}
		return goal;
	}

	/**
	 * Linear interpolation, also known as "Lerp", or "Mix". The method interpolates 
	 * within the range [start..end] based on a 't' parameter, where 't' is typically 
	 * within a [0..1] range.
	 * Examples:
	 * lerp(0, 100, 0.5); // 50
	 * lerp(20, 80, 0);   // 20
	 * 
	 * @param start the starting value
	 * @param end the ending value
	 * @param t percentage value in range 0..1 
	 */
	public static double lerp(double start, double end, double t) {
		// Precise method, which guarantees v = v1 when t = 1.
		  return (1 - t) * start + t * end;
	}
	/**
	 * Generate a random double between the supplied  ranges.
	 * 
	 * @param rangeMin start range
	 * @param rangeMax end range
	 * @return the random number
	 */
	public static double nextDouble(double rangeMin, double rangeMax) {
		return rangeMin + (rangeMax - rangeMin) * rnd.nextDouble();
	}

	/**
	 * Generate a random float between the supplied  ranges.
	 * 
	 * @param rangeMin start range
	 * @param rangeMax end range
	 * @return the random number
	 */
	public static double nextFloat(float rangeMin, float rangeMax) {
		return rangeMin + (rangeMax - rangeMin) * rnd.nextFloat();
	}
	
	/**
	 * Generate a random integer between the supplied  ranges.
	 * 
	 * @param rangeMin start range
	 * @param rangeMax end range
	 * @return the random number
	 */
	public static int nextInt(int rangeMin, int rangeMax) {
		return (int) (rangeMin + (rangeMax - rangeMin) * rnd.nextDouble());
	}
	
	/**
	 * Re-maps a number from one range to another. 
	 * 
	 * @param valueCoord1 The value to be converted
	 * @param startCoord1 Lower bound of the value's current range
	 * @param endCoord1 Upper bound of the value's current range
	 * @param startCoord2 Lower bound of the value's target range
	 * @param endCoord2 Upper bound of the value's target range
	 * @return The number re-mapped
	 */
	public static double map(double valueCoord1,
	        double startCoord1, double endCoord1,
	        double startCoord2, double endCoord2) {

	    if (Math.abs(endCoord1 - startCoord1) < EPSILON) {
	        throw new ArithmeticException("/ 0");
	    }

	    double offset = startCoord2;
	    double ratio = (endCoord2 - startCoord2) / (endCoord1 - startCoord1);
	    return ratio * (valueCoord1 - startCoord1) + offset;
	}
	/**
	 * Returns the Perlin noise value at the specified coordinate. Perlin noise is a random 
	 * sequence generator producing a more natural, harmonic succession of numbers than that 
	 * of the standard random function. It was developed by Ken Perlin in the 1980s and has 
	 * been used in graphical applications to generate procedural textures, shapes, terrains, and 
	 * other seemingly organic forms.  For more information see the
	 * <a href="https://en.wikipedia.org/wiki/Perlin_noise">Wikipedia page about Perlin noise</a>.
	 * 
	 * @param x x co-ordinate in noise space
	 * @return the computed noise value
	 * @see #noise(double)
	 * @see #noise(double, double)
	 */
	public static double noise(double x) {
		return noise(x, 0, 0);
	}
	/**
	 * Returns the Perlin noise value at specified coordinates. Perlin noise is a random 
	 * sequence generator producing a more natural, harmonic succession of numbers than that 
	 * of the standard random function. It was developed by Ken Perlin in the 1980s and has 
	 * been used in graphical applications to generate procedural textures, shapes, terrains, and 
	 * other seemingly organic forms.  For more information see the
	 * <a href="https://en.wikipedia.org/wiki/Perlin_noise">Wikipedia page about Perlin noise</a>. 
	 * 
	 * @param x x co-ordinate in noise space
	 * @param y y co-ordinate in noise space
	 * @return the computed noise value
	 * @see #noise(double)
	 * @see #noise(double, double)
	 */
	public static double noise(double x, double y) {
		return noise(x, y, 0);
	}
	
	/**
	 * Returns the Perlin noise value at specified coordinates. Perlin noise is a random 
	 * sequence generator producing a more natural, harmonic succession of numbers than that 
	 * of the standard random function. It was developed by Ken Perlin in the 1980s and has 
	 * been used in graphical applications to generate procedural textures, shapes, terrains, and 
	 * other seemingly organic forms. For more information see the
	 * <a href="https://en.wikipedia.org/wiki/Perlin_noise">Wikipedia page about Perlin noise</a>.
	 * 
	 * @param x x co-ordinate in noise space
	 * @param y y co-ordinate in noise space
	 * @param z z co-ordinate in noise space
	 * @return the computed noise value
	 * @see #noise(double)
	 * @see #noise(double, double)
	 */
	public static double noise(double x, double y, double z) {
		return PerlinNoise.noise(x, y, z);
	}
}
