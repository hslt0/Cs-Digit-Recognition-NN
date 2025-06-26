using DigitRecognitionNN.Utils;

namespace DigitRecognitionNN.Models;

public class Matrix
{
    private readonly double[] data;
    public int Rows { get; }
    public int Cols { get; }

    public double this[int row, int col]
    {
        get => data[row * Cols + col];
        set => data[row * Cols + col] = value;
    }

    public Matrix(int rows, int cols)
    {
        Rows = rows;
        Cols = cols;
        data = new double[rows * cols];
    }

    public Matrix(double[,] input)
    {
        Rows = input.GetLength(0);
        Cols = input.GetLength(1);
        data = new double[Rows * Cols];
        for (int i = 0; i < Rows; i++)
        for (int j = 0; j < Cols; j++)
            this[i, j] = input[i, j];
    }

    public Matrix Copy()
    {
        var result = new Matrix(Rows, Cols);
        Array.Copy(data, result.data, data.Length);
        return result;
    }

    public Matrix Transpose()
    {
        var result = new Matrix(Cols, Rows);
        for (int i = 0; i < Rows; i++)
        for (int j = 0; j < Cols; j++)
            result[j, i] = this[i, j];
        return result;
    }

    public void RandomizeWeights()
    {
        for (int i = 0; i < data.Length; i++)
            data[i] = MathUtils.RandomWeight();
    }

    public double[] ToArray() => data.ToArray();

    public static Matrix FromArray(double[] array)
    {
        var result = new Matrix(array.Length, 1);
        for (int i = 0; i < array.Length; i++)
            result[i, 0] = array[i];
        return result;
    }

    public double[][] ToJaggedArray()
    {
        var result = new double[Rows][];
        for (int i = 0; i < Rows; i++)
        {
            result[i] = new double[Cols];
            for (int j = 0; j < Cols; j++)
                result[i][j] = this[i, j];
        }
        return result;
    }

    public static Matrix FromJaggedArray(double[][] array)
    {
        int rows = array.Length;
        int cols = array[0].Length;
        var result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result[i, j] = array[i][j];
        return result;
    }

    public static Matrix operator +(Matrix a, Matrix b)
    {
        if (a.Rows != b.Rows || a.Cols != b.Cols)
            throw new InvalidOperationException("Matrices must have the same dimensions.");

        var result = new Matrix(a.Rows, a.Cols);
        for (int i = 0; i < a.data.Length; i++)
            result.data[i] = a.data[i] + b.data[i];
        return result;
    }

    public static Matrix operator -(Matrix a, Matrix b)
    {
        if (a.Rows != b.Rows || a.Cols != b.Cols)
            throw new InvalidOperationException("Matrices must have the same dimensions.");

        var result = new Matrix(a.Rows, a.Cols);
        for (int i = 0; i < a.data.Length; i++)
            result.data[i] = a.data[i] - b.data[i];
        return result;
    }

    public static Matrix operator *(Matrix a, double scalar)
    {
        var result = new Matrix(a.Rows, a.Cols);
        for (int i = 0; i < a.data.Length; i++)
            result.data[i] = a.data[i] * scalar;
        return result;
    }

    public static Matrix operator *(Matrix a, Matrix b)
    {
        if (a.Cols != b.Rows)
            throw new InvalidOperationException("A.Cols must equal B.Rows.");
        
        int aRows = a.Rows;
        int aCols = a.Cols;
        int bCols = b.Cols;

        var result = new Matrix(aRows, bCols);
        for (int i = 0; i < aRows; i++)
        {
            int aRowOffset = i * aCols;
            int rRowOffset = i * bCols;

            for (int j = 0; j < bCols; j++)
            {
                double sum = 0;
                for (int k = 0; k < aCols; k++)
                {
                    sum += a.data[aRowOffset + k] * b.data[k * bCols + j];
                }
                result.data[rRowOffset + j] = sum;
            }
        }
        return result;
    }
}