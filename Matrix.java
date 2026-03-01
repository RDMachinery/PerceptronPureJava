package org.anticml.math;

/**
 * A special purpose matrix maths class built for implementing machine
 * learning algorithms. This class may be suitable
 * for programs that make use of matrix computations and matrix
 * manipulations other than machine learning projects. 
 * Most of the
 * methods one would expect to find for machine learning applications are implemented. For example,
 * <code>add</code>,<code>subtract</code>,<code>multiply</code>
 * ,<code>divide</code>,<code>sum</code>,<code>square</code> and <code>map</code>. In addition
 * to the mathematical operations there are a number of methods provided
 * for inserting data into, querying and modifying the structure.
 * <p>
 * The example below creates two matrices and then multiplies them
 * together:<p>
 * <code>
 * // Construct the matrices
 * Matrix a = new Matrix();
 Matrix b = new Matrix();
 // Create some 2D array data
 double[][] aData = {{1,3,2},{4,0,1}};
 double[][] bData = {{1,3},{0,1},{5,2}};
 // Read data into the matrices
 a.readData(aData);
 b.readData(bData);
 // Perform the multiplication
 Matrix c = a.multiply(b);
 </code>
 *
 * @author Mario Gianota 29 June 2019 gianotamario@gmail.com
 */
public class Matrix {
    double[][] matrix;
    int rows;
    int columns;

    /**
     * * Used as an argument to the <code>getAsVector</code> method to
     * return a column vector. 
     */
    public static final int COLUMN_VECTOR = 0;
    /**
     * Used as an argument to the <code>getAsVector</code> method to
     * return a row vector. 
     * 
     */
    public static final int ROW_VECTOR = 1;

    /**
     * Creates a new matrix object the dimensions of which are zero and
     * with no internal data. Use the <code>readData</code> methods to
     * load data into the matrix.
     * @see #readData
     */
    public Matrix() {}

    /**
     * Creates a new matrix with the specified number of rows and
     * columns. All of the elements are initialised to zero.
     *
     * @param numRows Number of rows
     * @param numCols Number of columns
     */
    public Matrix(int numRows, int numCols) {
        if( numRows < 0 || numCols < 0)
            throw new IllegalArgumentException("Negative dimension: " + numRows + "x" + numCols);
        matrix = new double[numRows][numCols];
        rows = numRows;
        columns = numCols;
    }

    /**
     * Computes the dot product of this matrix by the supplied matrix <code>m</code> and 
     * returns the result. This matrix is left unchanged. If A is this matrix and B is the supplied
     * matrix <code>m</code>, then this method performs the following
     * matrix calculation:
     *          <code>A*B</code>
     *
     * @param m The matrix to multiply by
     * @return result of multiplying this matrix by m
     * @see #multiply(Matrix)
     */
    public Matrix dot(Matrix m) {
        double[][] rhsMatrix = m.getMatrixArray();

        // Number of columns in this matrix must match the number of rows
        // in matrix m
        if( columns != m.getRows() )
            throw new IllegalArgumentException("Can't multiply matrices. "+
                    "Number of columns in left hand side do not match number of rows in right handside: "+
                    columns + "!=" + m.getRows());

        // Dimension of result matrix is equal to the number of rows
        // in this matrix by the number of columns in the rhsMatrix
        double[][] result = new double[matrix.length][rhsMatrix[0].length];
        //System.out.println("Dimension of result matrix="+result.length+"x"+result[0].length);
        double value = 0;
        int row = 0;


        // For each column in rhs matrix
        for(int column=0; column<rhsMatrix[0].length; column++) {
            // For each row in lhs matrix
            for(int i=0; i<matrix.length; i++) {
                // For each column in lhs matrix
                for(int j=0; j<matrix[0].length; j++) {
                    value = value + matrix[i][j] * rhsMatrix[row][column];
                    //System.out.print(matrix[i][j]+"*"+rhsMatrix[row][column]+"+");
                    row++;
                }
                //System.out.println("="+value);
                result[i][column] = value;
                value = 0;
                row=0;
            }
        }

        Matrix rim = new Matrix();
        rim.readData(result);
        return rim;
    }
    /**
     * Multiplies this matrix element by element with the supplied matrix <code>m</code> and returns the result
     * of the multiplication. This matrix remains unchanged. The number of rows and columns in both matrices
     * must agree for this operation to succeed. 
     *
     * @param m The matrix to multiply by
     * @return result of multiplying this matrix by m
     * @see #dot(Matrix)
     */
    public Matrix multiply(Matrix m) {
        if( rows != m.getRows() || columns != m.getColumns() ) {
            throw new IllegalArgumentException("Can't perform element multiply. The number of rows and columns in both matrices must agree.");
        }

        Matrix result = copy();
        double[][] rim = result.getMatrixArray();

        for(int i=0; i<rows; i++) {
            for(int j=0; j<columns; j++) {
                rim[i][j] = getValueAt(i,j) * m.getValueAt(i,j);
            }
        }
        result.readData(rim);
        return result;
    }
    /**
     * Multiplies every component of this matrix by the value <code>n</code> and returns the
     * result.
     *
     * @param n Multiplier
     * @return the result
     */
    public Matrix multiply(double n) {
        int rows = getRows();
        int cols = getColumns();
        Matrix result = copy();
        double[][] rim = result.getMatrixArray();

        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                rim[i][j] = rim[i][j] * n;
            }
        }
        result.readData(rim);
        return result;
    }
    /**
     * Divides every number in the matrix by the value <code>n</code> and returns the
     * result.
     *
     * @param n Divisor
     * @return Division result
     */
    public Matrix divide(double n) {
    	if( n == 0 )
    		throw new IllegalArgumentException("Cannot divide matrix by zero.");
        int rows = getRows();
        int cols = getColumns();
        Matrix result = copy();
        double[][] rim = result.getMatrixArray();

        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                rim[i][j] = rim[i][j] / n;
            }
        }

        result.readData(rim);
        return result;
    }

    /**
     * Computes the square of this matrix, that is the matrix with
     * all numbers squared and returns the result.
     *
     * @return The result
     */
    public Matrix square() {
        int rows = getRows();
        int cols = getColumns();
        Matrix result = copy();
        double[][] rim = result.getMatrixArray();

        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                rim[i][j] = rim[i][j] * rim[i][j];
            }
        }

        result.readData(rim);
        return result;

    }
    /**
     * Sums all numbers in the matrix and returns the result.
     *
     * @return the sum
     */
    public double sum() {
        double sum = 0d;
        for(int i=0; i<rows; i++) {
            for(int j=0; j<columns; j++) {
                sum += matrix[i][j];
            }
        }
        return sum;
    }
    /**
     * Adds a matrix to this matrix and returns the result. The
     * supplied matrix must be of the same dimension for this
     * operation to succeed.
     *
     * @param m The matrix to add
     * @return the result
     */
    public Matrix add(Matrix m) {
        if( m.getRows() != rows || m.getColumns() != columns ) {
            throw new IllegalArgumentException("Cannot add matrices. "+
                    "Number of rows and columns do not match: "+
                    rows+"x"+columns +
                    " != " +
                    m.getRows() +"x" + m.getColumns());
        }
        Matrix result = copy();
        double[][] resultArray = result.getMatrixArray();
        double[][] rhsArray = m.getMatrixArray();

        for(int i=0; i<rows; i++ ) {
            for(int j=0; j<columns; j++) {
                resultArray[i][j] += rhsArray[i][j];
            }
        }
        return result;
    }
    /**
     * Adds <code>n</code> with every number in the matrix and
     * returns the result.
     *
     * @param n The number to add
     * @return The result
     */
    public Matrix add(double n) {
        int rows = getRows();
        int cols = getColumns();
        Matrix result = copy();
        double[][] rim = result.getMatrixArray();

        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                rim[i][j] = rim[i][j] + n;
            }
        }

        result.readData(rim);
        return result;

    }
    /**
     * Subtracts the supplied matrix <code>m</code> from the matrix. If the
     * dimensions of the supplied array do not match then an <code>IllegalArgumentException</code>
     * is thrown.
     *
     * @param m The matrix to subtract
     * @return The result
     */
    public Matrix subtract(Matrix m) {
        if( m.getRows() != rows || m.getColumns() != columns ) {
            throw new IllegalArgumentException("Cannot subtract matrices. "+
                    "Number of rows and columns do not match: "+
                    rows+"x"+columns +
                    " != " +
                    m.getRows() +"x" + m.getColumns());
        }
        Matrix result = copy();
        double[][] resultArray = result.getMatrixArray();
        double[][] rhsArray = m.getMatrixArray();

        for(int i=0; i<rows; i++ ) {
            for(int j=0; j<columns; j++) {
                resultArray[i][j] -= rhsArray[i][j];
            }
        }
        return result;
    }
    /**
     * Subtracts <code>n</code> from every number in the matrix and
     * returns the result.
     *
     * @param n The number to subtract
     * @return The result
     */
    public Matrix subtract(double n) {
        int rows = getRows();
        int cols = getColumns();
        Matrix result = copy();
        double[][] rim = result.getMatrixArray();

        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                rim[i][j] = rim[i][j] - n;
            }
        }

        result.readData(rim);
        return result;

    }

    /**
     * Returns the matrix's transpose.
     *
     * @return the transposed matrix
     */
    public Matrix transpose() {
        double ret[][] = new double[columns][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                ret[j][i] = matrix[i][j];
            }
        }
        Matrix m = new Matrix();
        m.readData(ret);
        return m;
        //return null;
    }

    /**
     * Returns the identity matrix in which the diagonal is all ones and
     * all other elements are zero.
     *
     * @return Identity matrix
     */
    public Matrix identity() {
        throw new RuntimeException("Not implemented.");
        //return null;

    }

    /**
     * Computes the inverse of the matrix.
     *
     * @return The inverse
     * @exception NonInvertibleMatrixException If the matrix is not square
     */
    public Matrix inverse() throws NonInvertibleMatrixException {
        if( ! isSquare() )
            throw new NonInvertibleMatrixException("Cannot invert a non square matrix.");
        int n = rows;
        double x[][] = new double[n][n];
        double b[][] = new double[n][n];
        int index[] = new int[n];

        for (int i=0; i<n; ++i)
            b[i][i] = 1;

        // Transform the matrix into an upper triangle
        gaussian(matrix, index);

        // Update the matrix b[i][j] with the ratios stored
        for (int i=0; i<n-1; ++i)
            for (int j=i+1; j<n; ++j)
                for (int k=0; k<n; ++k)
                    b[index[j]][k] -= matrix[index[j]][i]*b[index[i]][k];

        // Perform backward substitutions
        for (int i=0; i<n; ++i)
        {
            x[n-1][i] = b[index[n-1]][i]/matrix[index[n-1]][n-1];
            for (int j=n-2; j>=0; --j)
            {
                x[j][i] = b[index[j]][i];
                for (int k=j+1; k<n; ++k)
                {
                    x[j][i] -= matrix[index[j]][k]*x[k][i];
                }

                x[j][i] /= matrix[index[j]][j];

            }

        }
        Matrix m = new Matrix();
        m.readData(x);
        return m;
    }

    private void gaussian(double a[][], int index[]) {
        int n = index.length;
        double c[] = new double[n];

        // Initialize the index
        for (int i=0; i<n; ++i)
            index[i] = i;

        // Find the rescaling factors, one from each row
        for (int i=0; i<n; ++i) {
            double c1 = 0;
            for (int j=0; j<n; ++j) {
                double c0 = Math.abs(a[i][j]);
                if (c0 > c1) c1 = c0;
            }
            c[i] = c1;
        }



        // Search the pivoting element from each column

        int k = 0;

        for (int j=0; j<n-1; ++j)
        {
            double pi1 = 0;
            for (int i=j; i<n; ++i)
            {
                double pi0 = Math.abs(a[index[i]][j]);
                pi0 /= c[index[i]];
                if (pi0 > pi1)
                {
                    pi1 = pi0;
                    k = i;
                }
            }

            // Interchange rows according to the pivoting order
            int itmp = index[j];
            index[j] = index[k];
            index[k] = itmp;

            for (int i=j+1; i<n; ++i)
            {
                double pj = a[index[i]][j]/a[index[j]][j];

                // Record pivoting ratios below the diagonal
                a[index[i]][j] = pj;

                // Modify other elements accordingly
                for (int l=j+1; l<n; ++l)
                    a[index[i]][l] -= pj*a[index[j]][l];
            }

        }

    }
    
    /**
     * Apply a function using every element of the matrix and return
     * the result.
     * 
     * @param f The function to call
     * @return The result of applying the function
     * @see Function
     */
    public Matrix map(Function f) {
    	Matrix m = copy();
    	double[][] mData = m.getMatrixArray();
    	
    	for(int i=0; i<rows; i++) {
    		for(int j=0; j<columns; j++) {
    			double value = mData[i][j];
    			mData[i][j] = f.calculate(value);
    		}
    	}
    	m.readData(mData);
    	return m;
    }
    
    /**
     * Static method to apply a function using each element of the supplied matrix as
     * an argument to that function and return the result.
     * 
     * @param matrix The matrix
     * @param function The function to apply
     * @see Function
     */
    public static Matrix map(Matrix matrix, Function function) {
    	Matrix result = matrix.copy();
    	double[][] rData = result.getMatrixArray();
    	int rows = matrix.getRows();
    	int cols = matrix.getColumns();
    	
    	for(int i=0; i<rows; i++) {
    		for(int j=0; j<cols; j++) {
    			double value = rData[i][j];
    			rData[i][j] = function.calculate(value);
    		}
    	}
    	return result;
    }
    /**
     * Returns true if the dimension of this matrix forms a square.
     *
     * @return Squared or not
     */
    public boolean isSquare() {
        return getRows() == getColumns();
    }
    /**
     * Compares the supplied matrix <code>m</code> with the matrix
     * number by number. If all of the numbers are equal (and the
     * dimensions match) the method returns true and false otherwise.
     *
     * @param m The matrix to compare
     * @return Equal or not
     */
    public boolean equals(Matrix m) {
        if( rows != m.getRows() || columns != m.getColumns())
            return false;
        double[][] rhsArray = m.getMatrixArray();
        for(int i=0; i<rows; i++) {
            for(int j=0; j<columns; j++) {
                if( matrix[i][j] != rhsArray[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }
    /**
     * Returns a copy of the matrix.
     *
     * @return A copy of the matrix
     */
    public Matrix copy() {
        Matrix copy = new Matrix(rows, columns);
        copy.readData(matrix);
        return copy;
    }

    public Matrix getRow(int row) { throw new RuntimeException("Not implemented."); }
    
    /**
     * Returns the column of values located at the supplied column index. The returned
     * Matrix is a one column Matrix with the number of rows equal to the number of
     * rows in this Matrix.
     * 
     * @param column
     * @return The column as a one column Matrix
     */
    public Matrix getColumn(int column) {
    	if( column < 0 || column > columns )
    		throw new IllegalArgumentException("No such column: " +column);
        Matrix m = new Matrix(rows,1);
        for(int i=0; i<rows; i++) {
            m.setValueAt(i, 0, matrix[i][column]);
        }
        return m;
    }
    /**
     * Return the value at the given row and column index.
     *
     * @param row The row
     * @param column The column
     * @return The value
     * @see #getMatrixArray()
     */
    public double getValueAt(int row, int column) {
        if( row < 0 || row > getRows() ||
                column < 0 || column > getColumns())
            throw new ArrayIndexOutOfBoundsException("Row and/or column out of bounds: "
                    + row + "," + column);
        return matrix[row][column];
    }

    public void addRows(int numRows) {
        throw new RuntimeException("Not implemented.");
    }
    public void addColumns(int numColumns) {
        throw new RuntimeException("Not implemented.");
    }
    
    /**
     * Converts an array to a matrix. The returned matrix has all elements
     * of the supplied array in column 1.
     * 
     * @param array The array
     * @return the Matrix
     */
    public static Matrix fromArray(double[] vector) {
        Matrix m = new Matrix(vector.length,1);
        double[][] mData = m.getMatrixArray();
        for(int row=0; row<vector.length; row++) {
        	mData[row][0] = vector[row];
        }
        return m;
    }

    /**
     * Randomises every element in the matrix to a random number between
     * -1 and 1.
     */
    public void randomize() {
    	for(int i=0; i<rows; i++) {
    		for(int j=0; j<columns; j++) {
    			matrix[i][j] = Math.random() * 2 - 1;
    		}
    	}
    }

    /**
     * Sets every element in the matrix to 1.
     */
    public void ones() {
    	for(int i=0; i<rows; i++) {
    		for(int j=0; j<columns; j++) {
    			matrix[i][j] = 1;
    		}
    	}
    }
    /**
     * Sets every element in the matrix to 0.
     */
    public void zeros() {
    	for(int i=0; i<rows; i++) {
    		for(int j=0; j<columns; j++) {
    			matrix[i][j] = 0;
    		}
    	}
    }
    /**
     * Sets every element in the matrix equal to <code>x</code>.
     * @param x the value to fill the matrix
     */
    public void fill(double x) {
    	for(int i=0; i<rows; i++) {
    		for(int j=0; j<columns; j++) {
    			matrix[i][j] = x;
    		}
    	}
    }

    /**
     * Converts the matrix into an array and returns the array. The conversion
     * is a column ordered conversion. That is each row of this matrix is placed
     * into the array in the order in which they appear in the matrix.
     * 
     * @return the array
     */
    public double[] toArray() {
    	double[] result = new double[rows * columns];
    	int k = 0;
    	for(int i=0; i<rows; i++) {
    		for(int j=0; j<columns; j++) {
    			result[k] = matrix[i][j];
    			k++;
    		}
    	}
    	return result;
    }
    public void appendRight(Matrix m) {
        throw new RuntimeException("Not implemented.");
    }
    public void appendBottom(Matrix m) {
        throw new RuntimeException("Not implemented.");
    }



    /**
     * Sets the value of the number at <code>row</code>, <code>column</code>
     * equal to <code>n</code>.
     *
     * @param row The row index
     * @param column The column index
     * @param n The new value
     */
    public void setValueAt(int row, int column, double n) {
        if( row < 0 || row > getRows() -1 ||
                column < 0 || column > getColumns() -1 )
            throw new ArrayIndexOutOfBoundsException("Row and/or column out of bounds: "
                    + row + "," + column);
        
        matrix[row][column] = n;
    }
    /**
     * Reads the the values held in the supplied data
     * array into the matrix. Any data currently held by the matrix will be
     * lost. A new internal matrix array is created with the same dimensions
     * as <code>data</code> to contain the data. Note that the values are copied
     * in from <code>data</code> into
     * the matrix. No reference to <code>data</code> is held internally.
     *
     * @param data The data to read
     */
    public void readData(double data[][]) {
        int rows = data.length;
        int cols = data[0].length;

        matrix = new double[rows][cols];
        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                matrix[i][j] = data[i][j];
            }
        }
        this.rows = rows;
        this.columns = cols;
    }
    /**
     * Returns the internal values of the matrix. Note that this is mutable. Changes
     * to the returned array will alter the internal array.
     *
     * @return the internal array of values
     */
    public double[][] getMatrixArray() {
        return matrix;
    }
    /**
     * Returns the dimension of the matrix as a Matrix object. The first element contains the
     * number of rows and the second contains the number of columns.
     * @return the dimension
     */
    public Matrix getDimension() {
        Matrix m = new Matrix(1,2);
        double[][] array = m.getMatrixArray();
        array[0][0] = rows;
        array[0][1] = columns;
        return m;
    }
    /**
     * Return the number of rows.
     * @return rows
     */
    public int getRows() {
        return rows;
    }
    /**
     * Return the number of columns.
     * @return columns
     */
    public int getColumns() {
        return columns;
    }

    /**
     * Returns a printable, readable String representation of the
     * internal matrix.
     *  -       -
     * @return the matrix "string-ified"
     */
    public String toString() {
        StringBuffer sb = new StringBuffer();
        int rows = getRows();
        int cols = getColumns();
        // TODO: pad and right-align numbers in columns
        for(int i=0; i<rows; i++) {
            for( int j=0; j< cols; j++) {
                sb.append(matrix[i][j]+" ");
            }
            sb.append("\n");
        }

        return sb.toString();
    }
    
    /**
     * Prints the array to stdout.
     *
     */
    public void print() {
    	System.out.println(toString());
    }
    
    // TODO: REmove main method
    public static void main(String[] args) {
        Matrix a = new Matrix();
        Matrix b = new Matrix();
        double[][] aData = {{1,3,2},{4,0,1}};
        double[][] bData = {{1,3},{0,1},{5,2}};
        a.readData(aData);
        b.readData(bData);

        Matrix multResult = a.dot(b);

        System.out.println("Matrix A:");
        System.out.println(a);
        System.out.println("Matrix B:");
        System.out.println(b);
        System.out.println("A*B=");
        System.out.println(multResult);

        a = new Matrix();
        b = new Matrix();
        double[][] aData2 = {{1,3},{2,5}};
        double[][] bData2 = {{0,1},{3,2}};
        a.readData(aData2);
        b.readData(bData2);
        multResult = a.dot(b);

        System.out.println("Matrix A:");
        System.out.println(a);
        System.out.println("Matrix B:");
        System.out.println(b);
        System.out.println("A*B=");
        System.out.println(multResult);

        System.out.println("transpose(a) = ");
        System.out.println(a.transpose());

        Matrix cv = new Matrix();
        double[][] cvData = {{1,1},{2,2},{3,3},{4,4}};
        cv.readData(cvData);
        System.out.println("cv = ");
        System.out.println(cv);
        System.out.println("cv.transpose() = ");
        cv = cv.transpose();
        System.out.println(cv);
        cv = cv.transpose();
        System.out.println("cv.transpose() = ");
        System.out.println(cv);

        Matrix c = new Matrix();
        double[][] cData = {{1,2},{3,4}};
        c.readData(cData);

        System.out.println("C = " + c);
        try {
            System.out.println("c.inverse() =" );
            System.out.println(c.inverse());
        }catch(Exception ex) {
            System.out.println(ex.getMessage());
        }
    }

}