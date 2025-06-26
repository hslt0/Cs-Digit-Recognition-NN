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
        float max = input.Max();
        float sumExp = 0;
        float[] expValues = new float[input.Length];

        for (int i = 0; i < input.Length; i++)
        {
            expValues[i] = (float)Math.Exp(input[i] - max);
            sumExp += expValues[i];
        }

        for (int i = 0; i < input.Length; i++)
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
        int rows = m.Rows;
        int cols = m.Cols;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i, j] = ReLu(m[i, j]);
    }

    public static void ApplyReLuDerivative(Matrix z, Matrix delta)
    {
        int rows = z.Rows;
        int cols = z.Cols;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                delta[i, j] *= ReLuDerivative(z[i, j]);
    }
}