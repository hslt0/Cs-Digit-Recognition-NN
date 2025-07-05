using System.Numerics;
using DigitRecognitionNN.Utils;

namespace DigitRecognitionNN.Models;

public class Matrix
{
    private readonly float[] data;
    public int Rows { get; }
    public int Cols { get; }

    public float this[int row, int col]
    {
        get => data[row * Cols + col];
        set => data[row * Cols + col] = value;
    }

    public Matrix(int rows, int cols)
    {
        Rows = rows;
        Cols = cols;
        data = new float[rows * cols];
    }

    public Matrix(float[,] input)
    {
        Rows = input.GetLength(0);
        Cols = input.GetLength(1);
        data = new float[Rows * Cols];
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

    public float[] ToArray() => data.ToArray();

    public static Matrix FromArray(float[] array)
    {
        var result = new Matrix(array.Length, 1);
        for (int i = 0; i < array.Length; i++)
            result[i, 0] = array[i];
        return result;
    }

    public float[][] ToJaggedArray()
    {
        var result = new float[Rows][];
        for (int i = 0; i < Rows; i++)
        {
            result[i] = new float[Cols];
            for (int j = 0; j < Cols; j++)
                result[i][j] = this[i, j];
        }
        return result;
    }

    public static Matrix FromJaggedArray(float[][] array)
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

        int n = a.data.Length;
        int width = Vector<float>.Count;
        int i = 0;

        var result = new Matrix(a.Rows, a.Cols);

        for (; i <= n - width; i += width)
        {
            var va = new Vector<float>(a.data, i);
            var vb = new Vector<float>(b.data, i);
            (va + vb).CopyTo(result.data, i);
        }

        for (; i < n; i++)
            result.data[i] = a.data[i] + b.data[i];

        return result;
    }

    public static Matrix operator -(Matrix a, Matrix b)
    {
        int n = a.data.Length;
        int width = Vector<float>.Count;
        int i = 0;

        var result = new Matrix(a.Rows, a.Cols);

        for (; i <= n - width; i += width)
        {
            var va = new Vector<float>(a.data, i);
            var vb = new Vector<float>(b.data, i);
            (va - vb).CopyTo(result.data, i);
        }

        for (; i < n; i++)
            result.data[i] = a.data[i] - b.data[i];

        return result;
    }

    public static Matrix operator *(Matrix a, float scalar)
    {
        var result = new Matrix(a.Rows, a.Cols);

        int n = a.data.Length;
        int width = Vector<float>.Count;
        int i = 0;

        var vScalar = new Vector<float>(scalar);

        for (; i <= n - width; i += width)
        {
            var va = new Vector<float>(a.data, i);
            (va * vScalar).CopyTo(result.data, i);
        }

        for (; i < n; i++)
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
        int processorCount = Environment.ProcessorCount;
        int chunkSize = aRows / processorCount;

        var bT = b.Transpose();
        var result = new Matrix(aRows, bCols);

        Parallel.For(0, processorCount, i =>
        {
            int start = i * chunkSize;
            int end = (i == processorCount - 1) ? aRows : start + chunkSize;
        
            ProcessMatrixChunkSpan(a, bT, result, start, end, aCols, bCols);
        });

        return result;
    }

    private static void ProcessMatrixChunkSpan(Matrix a, Matrix bT, Matrix result, int startRow, int endRow, int aCols, int bCols)
    {
        for (int i = startRow; i < endRow; i++)
        {
            int aRowOffset = i * aCols;
            int rRowOffset = i * bCols;
            var aRowSpan = new Span<float>(a.data, aRowOffset, aCols);

            for (int j = 0; j < bCols; j++)
            {
                int bRowOffset = j * aCols;
                var bRowSpan = new Span<float>(bT.data, bRowOffset, aCols);
            
                float sum = ProcessVectorDotSpan(aRowSpan, bRowSpan);
                result.data[rRowOffset + j] = sum;
            }
        }
    }

    private static float ProcessVectorDotSpan(Span<float> a, Span<float> b)
    {
        int n = a.Length;
        int width = Vector<float>.Count;
        float sum = 0;
        int i = 0;
    
        for (; i <= n - width; i += width)
        {
            var va = new Vector<float>(a.Slice(i, width));
            var vb = new Vector<float>(b.Slice(i, width));
            sum += Vector.Dot(va, vb);
        }
        
        for (; i < n; i++)
            sum += a[i] * b[i];
        
        return sum;
    }
}