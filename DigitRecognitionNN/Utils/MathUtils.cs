namespace DigitRecognitionNN.Utils;

public static class MathUtils
{
    private static readonly Random Random = new();
    
    public static double RandomWeight() => Random.NextDouble() * 2 - 1; // [-1, 1]
    
    //Alternative for CrossEntropy
    
    /*public static double MeanSquaredError(double[] predicted, double[] actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Arrays must be the same length.");

        double sum = predicted.Select((t, i) => t - actual[i]).Sum(diff => diff * diff);

        return sum / predicted.Length;
    }*/
    
    public static double CrossEntropy(double[] predicted, double[] actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Arrays must be the same length.");

        double epsilon = 1e-12; // avoiding log(0)
        double sum = predicted.Select((t, i) => actual[i] * Math.Log(t + epsilon)).Sum();

        return -sum;
    }
    
    public static int ArgMax(double[] array)
    {
        if (array.Length == 0)
            throw new ArgumentException("Array is empty.");

        int maxIndex = 0;
        double maxValue = array[0];

        for (int i = 1; i < array.Length; i++)
        {
            if (!(array[i] > maxValue)) continue;
            
            maxValue = array[i];
            maxIndex = i;
        }

        return maxIndex;
    }

}