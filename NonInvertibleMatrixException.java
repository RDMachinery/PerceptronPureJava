package org.anticml.math;

/**
 * Indicates the condition that a <code>Matrix</code> is non invertible.
 */
public class NonInvertibleMatrixException extends Exception {
    private String message;

    public NonInvertibleMatrixException(String message) {
        super(message);
    }
}
