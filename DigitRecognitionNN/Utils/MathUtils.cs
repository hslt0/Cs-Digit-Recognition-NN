using System.Numerics.Tensors;

namespace DigitRecognitionNN.Utils;

public static class MathUtils
{
    public static float RandomWeight() => (float)Random.Shared.NextDouble() * 2 - 1; // [-1, 1]
    
    //Alternative for CrossEntropy
    
    /*public static float MeanSquaredError(float[] predicted, float[] actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Arrays must be the same length.");

        float sum = predicted.Select((t, i) => t - actual[i]).Sum(diff => diff * diff);

        return sum / predicted.Length;
    }*/
    
    public static float CrossEntropy(float[] predicted, float[] actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Arrays must be the same length.");

        var epsilon = 1e-12f; // avoiding log(0)
        var sum = 0f;
        for (var i = 0; i < predicted.Length; i++)
        {
            sum += (float)(actual[i] * Math.Log(predicted[i] + epsilon));
        }

        return -sum;
    }
    
    public static int ArgMax(float[] array)
    {
        return TensorPrimitives.IndexOfMax(array);
    }

}