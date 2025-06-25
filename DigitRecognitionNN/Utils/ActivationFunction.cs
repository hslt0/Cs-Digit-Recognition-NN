using DigitRecognitionNN.Models;

namespace DigitRecognitionNN.Utils;

public static class ActivationFunctions
{
    // Sigmoid for hidden (more classic way, relu is better tho)
    //public static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
    //public static double SigmoidDerivative(double x) => x * (1.0 - x);
    
    // Softmax for output
    public static double[] Softmax(double[] input)
    {
        double max = input.Max();
        double sumExp = 0;
        double[] expValues = new double[input.Length];

        for (int i = 0; i < input.Length; i++)
        {
            expValues[i] = Math.Exp(input[i] - max);
            sumExp += expValues[i];
        }

        for (int i = 0; i < input.Length; i++)
        {
            expValues[i] /= sumExp;
        }

        return expValues;
    }
    
    // ReLU for hidden
    private static double ReLu(double x) => Math.Max(0, x);
    private static double ReLuDerivative(double x) => x > 0 ? 1 : 0;

    public static void ApplyReLu(Matrix m)
    {
        for (int i = 0; i < m.Rows; i++)
        for (int j = 0; j < m.Cols; j++)
            m[i, j] = ReLu(m[i, j]);
    }

    public static void ApplyReLuDerivative(Matrix z, Matrix delta)
    {
        for (int i = 0; i < z.Rows; i++)
        for (int j = 0; j < z.Cols; j++)
            delta[i, j] *= ReLuDerivative(z[i, j]);
    }

}