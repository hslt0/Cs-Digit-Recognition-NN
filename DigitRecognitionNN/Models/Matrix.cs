using System.Numerics.Tensors;
using DigitRecognitionNN.Utils;

namespace DigitRecognitionNN.Models;

public class Matrix
{
    private readonly float[] _data;
    private int Rows { get; }
    private int Cols { get; }

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
        Buffer.BlockCopy(input, 0, _data, 0, _data.Length * sizeof(float));
    }
    
    public Span<float> AsSpan() => _data.AsSpan();

    public Matrix Copy()
    {
        var result = new Matrix(Rows, Cols);
        Array.Copy(_data, result._data, _data.Length);
        return result;
    }

    public Matrix Transpose()
    {
        var result = new Matrix(Cols, Rows);
        var rows = Rows;
        var cols = Cols;
        
        Parallel.For(0, rows, i =>
        {
            var inputOffset = i * cols;
            for (var j = 0; j < cols; j++)
            {
                result._data[j * rows + i] = _data[inputOffset + j];
            }
        });
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
        Array.Copy(array, result._data, array.Length);
        return result;
    }

    public float[][] ToJaggedArray()
    {
        var result = new float[Rows][];
        for (var i = 0; i < Rows; i++)
        {
            result[i] = new float[Cols];
            Array.Copy(_data, i * Cols, result[i], 0, Cols);
        }
        return result;
    }

    public static Matrix FromJaggedArray(float[][] array)
    {
        var rows = array.Length;
        var cols = array[0].Length;
        var result = new Matrix(rows, cols);
        for (var i = 0; i < rows; i++)
        {
            Array.Copy(array[i], 0, result._data, i * cols, cols);
        }
        return result;
    }

    public static Matrix operator +(Matrix a, Matrix b)
    {
        if (a.Rows != b.Rows || a.Cols != b.Cols)
            throw new InvalidOperationException("Matrices must have the same dimensions.");

        var result = new Matrix(a.Rows, a.Cols);
        TensorPrimitives.Add(a._data, b._data, result._data);
        return result;
    }

    public static Matrix operator -(Matrix a, Matrix b)
    {
        if (a.Rows != b.Rows || a.Cols != b.Cols)
            throw new InvalidOperationException("Matrices must have the same dimensions.");

        var result = new Matrix(a.Rows, a.Cols);
        TensorPrimitives.Subtract(a._data, b._data, result._data);
        return result;
    }

    public static Matrix operator *(Matrix a, float scalar)
    {
        var result = new Matrix(a.Rows, a.Cols);
        TensorPrimitives.Multiply(a._data, scalar, result._data);
        return result;
    }
    
    public static Matrix operator *(Matrix a, Matrix b)
    {
        if (a.Cols != b.Rows)
            throw new InvalidOperationException("A.Cols must equal B.Rows.");

        var aRows = a.Rows;
        var aCols = a.Cols;
        var bCols = b.Cols;

        var bT = b.Transpose();
        var result = new Matrix(aRows, bCols);

        Parallel.For(0, aRows, i =>
        {
            var aRowOffset = i * aCols;
            var rRowOffset = i * bCols;
            var aRowSpan = new ReadOnlySpan<float>(a._data, aRowOffset, aCols);

            for (var j = 0; j < bCols; j++)
            {
                var bRowOffset = j * aCols;
                var bRowSpan = new ReadOnlySpan<float>(bT._data, bRowOffset, aCols);
            
                var sum = TensorPrimitives.Dot(aRowSpan, bRowSpan);
                result._data[rRowOffset + j] = sum;
            }
        });

        return result;
    }
}