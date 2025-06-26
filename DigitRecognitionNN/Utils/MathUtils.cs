namespace DigitRecognitionNN.Utils;

public static class MathUtils
{
    private static readonly Random Random = new();
    
    public static float RandomWeight() => (float)Random.NextDouble() * 2 - 1; // [-1, 1]
    
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

        float epsilon = 1e-12f; // avoiding log(0)
        float sum = predicted.Select((t, i) => (float)(actual[i] * Math.Log(t + epsilon))).Sum();

        return -sum;
    }
    
    public static int ArgMax(float[] array)
    {
        if (array.Length == 0)
            throw new ArgumentException("Array is empty.");

        int maxIndex = 0;
        float maxValue = array[0];

        for (int i = 1; i < array.Length; i++)
        {
            if (!(array[i] > maxValue)) continue;
            
            maxValue = array[i];
            maxIndex = i;
        }

        return maxIndex;
    }

}