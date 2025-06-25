namespace DigitRecognitionNN.Models;

public class DataPoint
{
    public double[] Input { get; }    // 784 pixels (28x28)
    public int Label { get; }         // digit 0-9
    public double[] Target { get; }   // One-hot encoded [0,0,1,0,0,0,0,0,0,0]
    
    public DataPoint(double[] input, int label)
    {
        Input = input;
        Label = label;
        Target = CreateOneHot(label);
    }
    
    private double[] CreateOneHot(int label)
    {
        double[] result = new double[10];
        result[label] = 1.0;
        return result;
    }
}