using System.Numerics.Tensors;
using DigitRecognitionNN.Models;

namespace DigitRecognitionNN.Utils;

public static class ActivationFunctions
{
    // Sigmoid for hidden (more classic way, relu is better tho)
    //public static float Sigmoid(float x) => 1.0 / (1.0 + Math.Exp(-x));
    //public static float SigmoidDerivative(float x) => x * (1.0 - x);

    public static void ApplySoftmax(Matrix m)
    {
        TensorPrimitives.SoftMax(m.AsSpan(), m.AsSpan());
    }

    public static void ApplyReLu(Matrix m)
    {
        TensorPrimitives.Max(m.AsSpan(), 0f, m.AsSpan());
    }

    public static void ApplyReLuDerivative(Matrix z, Matrix delta)
    {
        var zSpan = z.AsSpan();
        var deltaSpan = delta.AsSpan();

        if (zSpan.Length != deltaSpan.Length)
            throw new ArgumentException("Matrices must have the same dimensions.");

        for (var i = 0; i < zSpan.Length; i++)
        {
            if (zSpan[i] <= 0)
                deltaSpan[i] = 0;
        }
    }
}