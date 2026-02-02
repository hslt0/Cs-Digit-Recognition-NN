using System.Numerics;
using DigitRecognitionNN.Utils;

namespace DigitRecognitionNN.Models;

public class Matrix
{
    private readonly float[] _data;
    public int Rows { get; }
    public int Cols { get; }

    public float this[int row, int col]
    {
        get => _data[row * Cols + col];
        set => _data[row * Cols + col] = value;
    }

    public Matrix(int rows, int cols)
    {
        Rows = rows;
        Cols = cols;
        _data = new float[rows * cols];
    }

    public Matrix(float[,] input)
    {
        Rows = input.GetLength(0);
        Cols = input.GetLength(1);
        _data = new float[Rows * Cols];
        for (var i = 0; i < Rows; i++)
        for (var j = 0; j < Cols; j++)
            this[i, j] = input[i, j];
    }

    public Matrix Copy()
    {
        var result = new Matrix(Rows, Cols);
        Array.Copy(_data, result._data, _data.Length);
        return result;
    }

    public Matrix Transpose()
    {
        var result = new Matrix(Cols, Rows);
        for (var i = 0; i < Rows; i++)
        for (var j = 0; j < Cols; j++)
            result[j, i] = this[i, j];
        return result;
    }

    public void RandomizeWeights()
    {
        for (var i = 0; i < _data.Length; i++)
            _data[i] = MathUtils.RandomWeight();
    }

    public float[] ToArray() => _data.ToArray();

    public static Matrix FromArray(float[] array)
    {
        var result = new Matrix(array.Length, 1);
        for (var i = 0; i < array.Length; i++)
            result[i, 0] = array[i];
        return result;
    }

    public float[][] ToJaggedArray()
    {
        var result = new float[Rows][];
        for (var i = 0; i < Rows; i++)
        {
            result[i] = new float[Cols];
            for (var j = 0; j < Cols; j++)
                result[i][j] = this[i, j];
        }
        return result;
    }

    public static Matrix FromJaggedArray(float[][] array)
    {
        var rows = array.Length;
        var cols = array[0].Length;
        var result = new Matrix(rows, cols);
        for (var i = 0; i < rows; i++)
        for (var j = 0; j < cols; j++)
            result[i, j] = array[i][j];
        return result;
    }

    public static Matrix operator +(Matrix a, Matrix b)
    {
        if (a.Rows != b.Rows || a.Cols != b.Cols)
            throw new InvalidOperationException("Matrices must have the same dimensions.");

        var n = a._data.Length;
        var width = Vector<float>.Count;
        var i = 0;

        var result = new Matrix(a.Rows, a.Cols);

        for (; i <= n - width; i += width)
        {
            var va = new Vector<float>(a._data, i);
            var vb = new Vector<float>(b._data, i);
            (va + vb).CopyTo(result._data, i);
        }

        for (; i < n; i++)
            result._data[i] = a._data[i] + b._data[i];

        return result;
    }

    public static Matrix operator -(Matrix a, Matrix b)
    {
        var n = a._data.Length;
        var width = Vector<float>.Count;
        var i = 0;

        var result = new Matrix(a.Rows, a.Cols);

        for (; i <= n - width; i += width)
        {
            var va = new Vector<float>(a._data, i);
            var vb = new Vector<float>(b._data, i);
            (va - vb).CopyTo(result._data, i);
        }

        for (; i < n; i++)
            result._data[i] = a._data[i] - b._data[i];

        return result;
    }

    public static Matrix operator *(Matrix a, float scalar)
    {
        var result = new Matrix(a.Rows, a.Cols);

        var n = a._data.Length;
        var width = Vector<float>.Count;
        var i = 0;

        var vScalar = new Vector<float>(scalar);

        for (; i <= n - width; i += width)
        {
            var va = new Vector<float>(a._data, i);
            (va * vScalar).CopyTo(result._data, i);
        }

        for (; i < n; i++)
            result._data[i] = a._data[i] * scalar;

        return result;
    }
    
    public static Matrix operator *(Matrix a, Matrix b)
    {
        if (a.Cols != b.Rows)
            throw new InvalidOperationException("A.Cols must equal B.Rows.");

        var aRows = a.Rows;
        var aCols = a.Cols;
        var bCols = b.Cols;
        var processorCount = Environment.ProcessorCount;
        var chunkSize = aRows / processorCount;

        var bT = b.Transpose();
        var result = new Matrix(aRows, bCols);

        Parallel.For(0, processorCount, i =>
        {
            var start = i * chunkSize;
            var end = (i == processorCount - 1) ? aRows : start + chunkSize;
        
            ProcessMatrixChunkSpan(a, bT, result, start, end, aCols, bCols);
        });

        return result;
    }

    private static void ProcessMatrixChunkSpan(Matrix a, Matrix bT, Matrix result, int startRow, int endRow, int aCols, int bCols)
    {
        for (var i = startRow; i < endRow; i++)
        {
            var aRowOffset = i * aCols;
            var rRowOffset = i * bCols;
            var aRowSpan = new Span<float>(a._data, aRowOffset, aCols);

            for (var j = 0; j < bCols; j++)
            {
                var bRowOffset = j * aCols;
                var bRowSpan = new Span<float>(bT._data, bRowOffset, aCols);
            
                var sum = ProcessVectorDotSpan(aRowSpan, bRowSpan);
                result._data[rRowOffset + j] = sum;
            }
        }
    }

    private static float ProcessVectorDotSpan(Span<float> a, Span<float> b)
    {
        var n = a.Length;
        var width = Vector<float>.Count;
        float sum = 0;
        var i = 0;
    
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