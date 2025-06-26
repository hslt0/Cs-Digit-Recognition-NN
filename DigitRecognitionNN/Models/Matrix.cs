using DigitRecognitionNN.Utils;

namespace DigitRecognitionNN.Models;

public class Matrix
{
    private readonly float[,] data;
    public int Rows { get; }
    public int Cols { get; }
    
    // Constructors
    public Matrix(int rows, int cols)
    {
        Rows = rows;
        Cols = cols;
        data = new float[rows, cols];
    }

    public Matrix(float[,] data)
    {
        this.data = data;
        Rows = data.GetLength(0);
        Cols = data.GetLength(1);
    }
    
    // Operations with matrices
    public static Matrix operator +(Matrix a, Matrix b)
    {
        if (a.Rows != b.Rows || a.Cols != b.Cols)
            throw new InvalidOperationException("Matrices must have the same dimensions for addition.");

        var result = new Matrix(a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < a.Cols; j++)
            {
                result[i, j] = a[i, j] + b[i, j];
            }
        }
        return result;
    }

    public static Matrix operator -(Matrix a, Matrix b)
    {
        if (a.Rows != b.Rows || a.Cols != b.Cols)
            throw new InvalidOperationException("Matrices must have the same dimensions for subtraction.");

        Matrix result = new Matrix(a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < a.Cols; j++)
            {
                result[i, j] = a[i, j] - b[i, j];
            }
        }
        return result;
    }

    public static Matrix operator *(Matrix a, Matrix b)
    {
        if (a.Cols != b.Rows)
            throw new InvalidOperationException("Number of columns of A must equal number of rows of B.");

        Matrix result = new Matrix(a.Rows, b.Cols);
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < b.Cols; j++)
            {
                float sum = 0;
                for (int k = 0; k < a.Cols; k++)
                {
                    sum += a[i, k] * b[k, j];
                }
                result[i, j] = sum;
            }
        }
        return result;
    }

    public static Matrix operator *(Matrix a, float b)
    {
        var result = new Matrix(a.Rows, a.Cols);
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < a.Cols; j++)
            {
                result[i, j] = a[i, j] * b;
            }
        }
        return result;
    }

    public Matrix Transpose()
    {
        var result = new Matrix(Cols, Rows);
        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < Cols; j++)
            {
                result[j, i] = this[i, j];
            }
        }
        return result;
    }

    
    // Init
    public void RandomizeWeights()
    {
        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < Cols; j++)
            {
                data[i, j] = MathUtils.RandomWeight();
            }
        }
    }
    
    public Matrix Copy()
    {
        var result = new Matrix(Rows, Cols);
        for (int i = 0; i < Rows; i++)
        for (int j = 0; j < Cols; j++)
            result[i, j] = this[i, j];
        return result;
    }

    //public void SetZero() => Array.Clear(data, 0, data.Length);
    
    // Indexer
    public float this[int row, int col]
    {
        get => data[row, col];
        set => data[row, col] = value;
    }

    
    // Transform
    public float[] ToArray()
    {
        float[] result = new float[Rows * Cols];
        int k = 0;
        
        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < Cols; j++)
            {
                result[k] = this[i, j];
                k++;
            }
        }
        
        return result;
    }

    public static Matrix FromArray(float[] array)
    {
        var result = new Matrix(array.Length, 1);
        for (int i = 0; i < array.Length; i++)
        {
            result[i, 0] = array[i];
        }
        return result;
    }

    public float[][] ToJaggedArray()
    {
        float[][] result = new float[Rows][];
        for (int i = 0; i < Rows; i++)
        {
            result[i] = new float[Cols];
            for (int j = 0; j < Cols; j++)
                result[i][j] = this[i, j];
        }
        return result;
    }
    public static Matrix FromJaggedArray(float[][]? array)
    {
        if (array == null)
            throw new ArgumentNullException(nameof(array), "Array cannot be null.");

        int rows = array.Length;
        int cols = array[0].Length;

        var matrix = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i, j] = array[i][j];

        return matrix;
    }
}