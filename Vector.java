package org.anticml.math;
/**
 * A vector implementation for vector maths. 
 * 
 * @author Mario
 *
 */
public class Vector {
	  public double x;
	  public double y;
	  public double z;

	  /**
	   * Constructs a new vector with values initialised to zero
	   */
	  public Vector() {
	  }

	  /**
	   * Constructs a new vector from 3 values
	   * @param x the x value
	   * @param y the y value
	   * @param z the z value
	   */
	  public Vector(double x, double y, double z) {
	    this.x = x;
	    this.y = y;
	    this.z = z;
	  }

	  /**
	   * Constructs a Vector from 2 points
	   * @param x the x value
	   * @param y the y value
	   */
	  public Vector(double x, double y) {
		  this.x = x;
		  this.y = y;
		  this.z = 0;
	  }
	  
	  /**
	   * Set the Vector's values
	   * @param x the x value to set
	   * @param y the y value to set
	   * @param z the z value to set
	   */
	  public void set(double x, double y, double z) {
		  this.x = x;
		  this.y = y;
		  this.z = z;
	  }
	  /**
	   * Set the Vector's values
	   * @param x the x value to set
	   * @param y the y value to set
	   */
	  public void set(double x, double y) {
		  this.x = x;
		  this.y = y;
	  }
	  /**
	   * Zeros out the x,y and z values of the Vector.
	   */
	  public void zeros() {
		  this.x = 0;
		  this.y = 0;
		  this.z = 0;
	  }
	  /**
	   * Add a Vector to this Vector
	   * @param arg the Vector to add to this
	   * @return the result of the addition (this vector)
	   */
	  public Vector add(Vector arg) {
		  x += arg.x;
		  y += arg.y;
		  z += arg.z;
		  return this;
	  }
	  /**
	   * Add Vector a to Vector b
	   * @param a left handside of addition
	   * @param b right handside of addition
	   * @return the result of the addition 
	   */
	  public static Vector add(Vector a, Vector b) {
		  double x = a.x + b.x;
		  double y = a.y + b.y;
		  double z = a.z + b.z;
		  return new Vector(x,y,z);
	  }
	  
	  /**
	   * Subtract a Vector from this Vector
	   * @param arg the Vector to subtract from this
	   * @return the result of the subtraction (this vector)
	   */
	  public Vector sub(Vector arg) {
		  x -= arg.x;
		  y -= arg.y;
		  z -= arg.z;
		  return this;
	  }
	  /**
	   * Subtract a number from this Vector
	   * @param arg the number to subtract from this
	   * @return the result of the subtraction (this vector)
	   */
	  public Vector sub(double arg) {
		  x -= arg;
		  y -= arg;
		  z -= arg;
		  return this;
	  }
	  /**
	   * Subtract Vector b from Vector a
	   * @param a left handside of subtraction
	   * @param b right handside of subtraction
	   * @return the result of the subtraction 
	   */
	  public static Vector sub(Vector a, Vector b) {
		  double x = a.x - b.x;
		  double y = a.y - b.y;
		  double z = a.z - b.z;
		  return new Vector(x,y,z);
	  }

	  /**
	   * Multiply every element of the Vector by <em>arg</em>
	   * @param arg
	   * @return the result of the scalar multiplication (this Vector)
	   */
	  public Vector mult(double arg) {
		  x *= arg;
		  y *= arg;
		  z *= arg;
		  return this;
	  }
	  
	  /**
	   * Divide every element of the Vector by <em>arg</em>
	   * @param arg
	   * @return the result of the scalar division (this Vector)
	   */
	  public Vector div(double arg) {
		  x /= arg;
		  y /= arg;
		  z /= arg;
		  return this;
	  }
	  /**
	   * Calculates the magnitude (length) of the vector and returns the result
	   * as a double (this is simply the equation <em>sqrt(x*x + y*y + z*z)</em>.)
	   *
	   * @return magnitude (length) of the vector
	   * @see Vector#magSq()
	   */
	  public double mag() {
	    return (double) Math.sqrt(x*x + y*y + z*z);
	  }
	  
	  /**
	   *
	   * Calculates the squared magnitude of the vector and returns the result
	   * as a double (this is simply the equation <em>(x*x + y*y + z*z)</em>.)
	   * Faster if the real length is not required in the
	   * case of comparing vectors, etc.
	   *
	   * @return squared magnitude of the vector
	   * @see Vector#mag()
	   */
	  public double magSq() {
	    return (x*x + y*y + z*z);
	  }
	  
	  /**
	   * Returns a copy of this vector.
	   * 
	   * @return the copy
	   */
	  public Vector copy() {
		  Vector v = new Vector(x,y,z);
		  return v;
	  }
	  
	  /**
	   * Normalize this vector.
	   * 
	   * @return This vector normalized
	   */
	  public Vector normalize() {
		  double m = mag();
		  if( m != 0 && m != 1)
			  return this.div(m);
		  return this;
	  }
	  
	  /**
	   * Make a new 2D unit vector from an angle.
	   * 
	   * @param angle The angle in radians
	   * @return the new unit vector
	   */
	  public static Vector fromAngle(double angle) {
		    return fromAngle(angle,null);
	  }
	  
	  /**
	   * Make a new 2D unit vector from an angle
	   *
	   * @param target the target vector (if null, a new vector will be created)
	   * @return the Vector
	   */
	  public static Vector fromAngle(double angle, Vector target) {
	    if (target == null) {
	      target = new Vector((double)Math.cos(angle),(double)Math.sin(angle),0);
	    } else {
	      target.set((double)Math.cos(angle),(double)Math.sin(angle),0);
	    }
	    return target;
	  }
	  
	  /**
	   * Calculates the Euclidean distance between two points (considering a
	   * point as a vector object).
	   *
	   * @param v the x, y, and z coordinates of a Vector
	   */
	  public double dist(Vector v) {
	    double dx = x - v.x;
	    double dy = y - v.y;
	    double dz = z - v.z;
	    return (double)Math.sqrt(dx*dx + dy*dy + dz*dz);
	  }
	  /**
	   * Calculates the Euclidean distance between two points (considering a
	   * point as a vector object).
	   *
	   * @param v1 the x, y, and z coordinates of a Vector
	   * @param v2 the x, y and z coordinates of a Vector
	   */
	  public static double dist(Vector v1, Vector v2) {
	    double dx = v2.x - v1.x;
	    double dy = v2.y - v1.y;
	    double dz = v2.z - v1.z;
	    return (double)Math.sqrt(dx*dx + dy*dy + dz*dz);
	  }
	  
	  /**
	   * Calculates the dot product of two vectors.
	   * @param v any variable of type Vector
	   * @return the dot product
	   */
	  public double dot(Vector v) {
	    return x*v.x + y*v.y + z*v.z;
	  }
	  /**
	   * Calculates and returns a new vector composed of the cross product between
	   * two vectors.
	   * @param v the vector to calculate the cross product
	   * @return The cross product
	   */
	  public Vector cross(Vector v) {
	    return cross(v, null);
	  }

	  /**
	   * @param v any variable of type Vector
	   * @param target Vector to store the result
	   */
	  public Vector cross(Vector v, Vector target) {
	    double crossX = y * v.z - v.y * z;
	    double crossY = z * v.x - v.z * x;
	    double crossZ = x * v.y - v.x * y;

	    if (target == null) {
	      target = new Vector(crossX, crossY, crossZ);
	    } else {
	      target.set(crossX, crossY, crossZ);
	    }
	    return target;
	  }
	  
	  /**
	   * Limit the magnitude of this vector to the value used for the <b>max</b> parameter.
	   *
	   * @param max the maximum magnitude for the vector
	   */
	  public Vector limit(double max) {
	    if (magSq() > max*max) {
	      normalize();
	      mult(max);
	    }
	    return this;
	  }
	  
	  /**
	   * Set the magnitude of this vector to the value used for the <b>len</b> parameter.
	   *
	   * @param len the new length for this vector
	   */
	  public Vector setMag(double len) {
	    normalize();
	    mult(len);
	    return this;
	  }
	  
	  /**
	   * Calculate the angle of rotation for this vector (only 2D vectors)
	   * @return the angle of rotation
	   */
	  public double heading() {
	    double angle = (double)Math.atan2(y, x);
	    return angle; 
	  }
	  
	  /**
	   * Creates a random Vector
	   * @return the random vector
	   */
	  public static Vector random() {
		  
		  return fromAngle(Math.random() * Math.PI * 2);
	  }
	  
	  /**
	   * Creates a copy of this Vector and returns it.
	   * @return vector copy
	   */
	  public Vector clone() {
		  return new Vector(x,y,z);
	  }
	  
	  public String toString() {
		  return x+","+y+","+z;
	  }
}