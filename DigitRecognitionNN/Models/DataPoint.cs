namespace DigitRecognitionNN.Models;

public class DataPoint
{
    public float[] Input { get; }    // 784 pixels (28x28)
    public int Label { get; }         // digit 0-9
    public float[] Target { get; }   // One-hot encoded [0,0,1,0,0,0,0,0,0,0]
    
    public DataPoint(float[] input, int label)
    {
        Input = input;
        Label = label;
        Target = CreateOneHot(label);
    }
    
    private float[] CreateOneHot(int label)
    {
        float[] result = new float[10];
        result[label] = 1.0f;
        return result;
    }
}