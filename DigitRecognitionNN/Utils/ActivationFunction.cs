using DigitRecognitionNN.Models;

namespace DigitRecognitionNN.Utils;

public static class ActivationFunctions
{
    // Sigmoid for hidden (more classic way, relu is better tho)
    //public static float Sigmoid(float x) => 1.0 / (1.0 + Math.Exp(-x));
    //public static float SigmoidDerivative(float x) => x * (1.0 - x);
    
    // Softmax for output
    public static float[] Softmax(float[] input)
    {
        var max = input.Max();
        float sumExp = 0;
        var expValues = new float[input.Length];

        for (var i = 0; i < input.Length; i++)
        {
            expValues[i] = (float)Math.Exp(input[i] - max);
            sumExp += expValues[i];
        }

        for (var i = 0; i < input.Length; i++)
        {
            expValues[i] /= sumExp;
        }

        return expValues;
    }
    
    // ReLU for hidden
    private static float ReLu(float x) => Math.Max(0, x);
    private static float ReLuDerivative(float x) => x > 0 ? 1 : 0;

    public static void ApplyReLu(Matrix m)
    {
        var rows = m.Rows;
        var cols = m.Cols;
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                m[i, j] = ReLu(m[i, j]);
    }

    public static void ApplyReLuDerivative(Matrix z, Matrix delta)
    {
        var rows = z.Rows;
        var cols = z.Cols;
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                delta[i, j] *= ReLuDerivative(z[i, j]);
    }
}